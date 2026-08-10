"""One-shot environment certification for this repo on Oscar (or any container+venv setup).

Run it after ANY environment change -- fresh setup, git pull, xformers rebuild, cluster
upgrade -- and before every campaign. It encodes every failure class hit in practice:
the mislabeled container, the ~/.local user-site leak (shadowed torch/lgatr, venv holes),
wheel/ABI mismatches, crippled no-kernel xformers builds, GPU-less allocations.

    # CPU context (login node / CPU interact): provenance + stack checks only
    apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
        'source venv/bin/activate && python utils/env_check.py'

    # GPU context (interact with -g 1, apptainer --nv): everything, incl. real kernels
    apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc \
        'source venv/bin/activate && python utils/env_check.py --gpu'

Exit code 0 = certified. Any FAIL prints what broke and how it usually got that way.
"""
#needed because OSCAR is so irritating maybe im not used to env work but holy

import argparse
import glob
import importlib
import os
import sys

RESULTS = []


# Known attention backend names, for the lgatr probe. Kept as a literal because lgatr 2.0
# exposes no enumeration API -- `get_attention_backend` resolves ONE name at a time. An
# unknown name is distinguishable from an unavailable one (different ValueError text), so a
# stale entry here surfaces loudly instead of quietly shrinking the reported set.
_LGATR_BACKENDS = ("native", "varlen", "xformers", "flex", "flash")


def _lgatr_backends():
    """Backends lgatr 2.0 can actually bind, by asking it one name at a time.

    lgatr 1.4.4 kept a module-level `_REGISTRY` populated at import; 2.0 replaced that with
    LAZY resolution (`get_attention_backend` -> `_resolve_backend`, imported on first use),
    so there is no longer anything to enumerate and `_REGISTRY` does not exist. Probing the
    resolver is not merely the replacement idiom -- it is a better test, because it is the
    exact code path a model takes at its first forward, ABI failure included.

    Raises if the resolver itself is gone, so a third API shape cannot silently read as
    "no backends".
    """
    from lgatr.primitives.attention import get_attention_backend

    out = []
    for name in _LGATR_BACKENDS:
        try:
            get_attention_backend(backend=name)
            out.append(name)
        except ValueError as e:
            if "Unknown attention backend" in str(e):
                raise RuntimeError(
                    f"lgatr no longer knows the backend name {name!r} -- _LGATR_BACKENDS in "
                    f"env_check.py is stale against this lgatr ({e})") from e
            # "is not available": genuinely unbound here (no CUDA, missing wheel, bad ABI).
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"probing lgatr backend {name!r} raised "
                               f"{type(e).__name__}: {e}") from e
    return sorted(out)


def _lloca_backends():
    """lloca still uses the eager import-time registry, so read it -- but require it to
    EXIST. A missing `_REGISTRY` means lloca changed shape the way lgatr did, and the honest
    answer is then "unknown", not "empty"."""
    mod = importlib.import_module("lloca.backbone.attention_backends.mask")
    reg = getattr(mod, "_REGISTRY", None)
    if not isinstance(reg, dict):
        raise RuntimeError(
            f"lloca.backbone.attention_backends.mask._REGISTRY is {type(reg).__name__}, not a "
            f"dict -- lloca has moved to lazy backend resolution like lgatr 2.0; give it a "
            f"resolver-probe here, the way _lgatr_backends() does")
    return sorted(reg)


def check(name, ok, detail="", hint=""):
    RESULTS.append(ok)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
    if not ok and hint:
        print(f"       -> {hint}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="assert CUDA + run real GPU kernels")
    args = ap.parse_args()

    import socket
    host = socket.gethostname()
    print(f"context: {host}" + ("  (login node -- a --gpu run here is EXPECTED to fail "
                                "the CUDA check; use interact -g 1)" if host.startswith("login") else ""))

    # ---- 1. interpreter provenance ------------------------------------------------
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    check("venv interpreter", in_venv, sys.executable,
          "activate the venv inside the container: source venv/bin/activate")

    local_paths = [p for p in sys.path if "/.local/" in p]
    check("no ~/.local on sys.path", not local_paths, "; ".join(local_paths),
          "user-site leak: mv ~/.local/lib/pythonX.Y aside + export PYTHONNOUSERSITE=1")
    check("PYTHONNOUSERSITE set", os.environ.get("PYTHONNOUSERSITE") == "1",
          hint="export PYTHONNOUSERSITE=1 (persist in ~/.bashrc AND sbatch templates)")
    leaked = [d for d in glob.glob(os.path.expanduser("~/.local/lib/python3*"))
              if not d.endswith(".leaked")]
    check("no live ~/.local/lib trees", not leaked, "; ".join(leaked),
          "mv each aside (reversible): mv <dir> <dir>.leaked")
    pp = os.environ.get("PYTHONPATH", "")
    check("PYTHONPATH empty", not pp, pp,
          "something is force-injecting import paths past the venv -- find and remove "
          "the export (bashrc/profile/module)")
    conda_dirs = [d for d in ("~/miniconda3", "~/anaconda3", "~/.conda")
                  if os.path.isdir(os.path.expanduser(d))]
    pip_confs = [f for f in ("~/.config/pip/pip.conf", "~/.pip/pip.conf")
                 if os.path.isfile(os.path.expanduser(f))]
    if conda_dirs or pip_confs:
        print(f"[WARN] dormant installers present: {conda_dirs + pip_confs} -- harmless "
              "while inactive, but a conda init block or user pip.conf can redirect "
              "installs; audit if imports ever surprise you")

    # ---- 2. torch provenance + CUDA ----------------------------------------------
    import torch
    tfile = torch.__file__
    check("torch is the container build", "nv" in torch.__version__ and "/.local/" not in tfile,
          f"{torch.__version__} @ {tfile}",
          "a plain pip version string (e.g. 2.11.0+cu130) or a ~/.local path = the leak; "
          "NGC builds carry an nv tag and live under /usr/local")
    if args.gpu:
        ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if ok else ""
        check("CUDA available", ok, name,
              "on a GPU node with --nv? interact needs -g 1 -- an unallocated GPU is "
              "cgroup-hidden (cuda False with no error)")
        if ok:
            x = torch.randn(64, 64, device="cuda")
            check("GPU matmul executes", bool(((x @ x).sum()).isfinite().item()))
    else:
        print("[SKIP] CUDA checks (run with --gpu on a GPU allocation for full certification)")

    # ---- 3. science stack ---------------------------------------------------------
    import numpy, torch_geometric, lgatr, lloca, requests, tqdm  # noqa: F401
    check("numpy < 2", numpy.__version__.split(".")[0] == "1", numpy.__version__,
          "repo pins numpy<2 (weaver); a 2.x here means the wrong package won resolution")
    check("torch-geometric >= 2.6",
          tuple(int(v) for v in torch_geometric.__version__.split(".")[:2]) >= (2, 6),
          torch_geometric.__version__)
    lg_ok = "/venv/" in lgatr.__file__ or in_venv and sys.prefix in lgatr.__file__
    check("lgatr resolves from the venv", lg_ok,
          f"{lgatr.__version__} @ {lgatr.__file__}",
          "a non-venv path means a shadowing copy (the leak carried an older lgatr); "
          "re-run the section-2 install block")

    # ---- 3b. storage layout ---------------------------------------------------------
    # `runs/` and the full dataset must live OUTSIDE home. Home is 100 GB with an inode
    # quota shared with the venv, and it is snapshotted -- so deleting an overflowing runs/
    # does not free the quota immediately. The failure mode is a campaign that dies days in
    # with "Disk quota exceeded" and takes git and the venv down with it.
    #
    # This is checked because the documented `ln -sfn <target> runs` SILENTLY does the wrong
    # thing when `runs` already exists as a real directory: it creates `runs/<basename>`
    # INSIDE it and exits 0, leaving output in home. `data/toptagging_full.npz` is a file, so
    # its `ln -sfn` is unaffected -- which is why one of the two usually looks right.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    home = os.path.realpath(os.path.expanduser("~"))
    for rel in ("runs", "data/toptagging_full.npz", "data/JetClass", "data/toptagxl"):
        path = os.path.join(repo, rel)
        if not os.path.exists(path) and not os.path.islink(path):
            continue  # not set up yet (JetClass/TopTagXL are optional) -- nothing to judge
        is_link = os.path.islink(path)
        target = os.path.realpath(path)
        outside_home = not target.startswith(home + os.sep)
        nested = is_link is False and os.path.isdir(path) and any(
            os.path.islink(os.path.join(path, e)) for e in os.listdir(path)
        )
        check(f"{rel} lives outside home", is_link and outside_home,
              f"{'symlink -> ' if is_link else 'REAL DIRECTORY in home: '}{target}"
              + ("  (contains a nested symlink -- the `ln -sfn` footgun)" if nested else ""),
              "fix: mkdir -p <target>; "
              "find " + rel + " -mindepth 1 -maxdepth 1 -exec mv -t <target> {} + ; "
              "rmdir " + rel + " ; ln -sfn <target> " + rel + " ; then `ls -ld " + rel + "` "
              "MUST print an arrow. See docs/OSCAR.md section 2.")

    # whether a GPU is visible at all -- distinct from the --gpu flag, which asks for the
    # heavier kernel checks. Used to skip checks that are CUDA-gated by construction.
    cuda_here = torch.cuda.is_available()

    # ---- 4. xformers --------------------------------------------------------------
    # Absence is NOT a failure: the documented Oscar image is built xformers-free
    # (OSCAR.md section 2 strips it), and nothing in the campaign needs it -- only the four
    # configs that pin `attention_backend: xformers` do. Failing here would make the
    # RECOMMENDED setup permanently "NOT CERTIFIED", which trains people to ignore the tool.
    try:
        import xformers
    except ImportError:
        print("[INFO] xformers not installed -- expected on the xformers-free image. Only "
              "tag_transformer / tag_top_transformer / tag_lgatr / tag_slim pin it; override "
              "with model.attention_backend=native|flex (OSCAR.md).")
        xformers = None
    except Exception as e:  # noqa: BLE001 -- e.g. OSError from a broken .so, or xformers'
        # own version-gate RuntimeError. Reported as a FAIL, never as a traceback that would
        # kill the run before the summary line.
        check("xformers imports at all", False, f"{type(e).__name__}: {e}",
              "xformers is installed but its import raises. See the flash decision tree in "
              "docs/OSCAR.md section 2.2 -- an unimportable flash_attn crashes ALL of xformers.")
        xformers = None
    if xformers is not None:
        try:
            ver = xformers.__version__
            check("xformers real build (sha-stamped)", "+" in ver, ver,
                  "a bare version = the crippled kernel-free sdist build; rebuild from the "
                  "git tag matched to the container torch (see OSCAR.md)")
            import xformers.ops as xops  # the import that the flash chain can crash
            check("xformers.ops imports (flash chain resolved)", True)

            # Present != functional, and the difference is invisible from `import xformers`.
            # `xformers._cpp_lib` records the load result of the compiled extension: None when
            # the C++/CUDA kernels bound to this torch, an exception object when they did not.
            # That is the authoritative signal and it works on CPU, so a login-node run already
            # tells you whether the build is real.
            # (Do NOT test this by importing `xformers._C`: it is a torch library loaded via
            # torch.ops, not a Python extension module, so a healthy build raises
            # "does not define module export function (PyInit__C)" -- a false alarm.)
            try:
                from xformers import _cpp_lib
                exc = _cpp_lib._cpp_library_load_exception
                check("xformers C++/CUDA extensions loaded", exc is None,
                      "clean" if exc is None else f"{type(exc).__name__}: {str(exc)[:120]}",
                      "the wheel's kernels cannot bind to this torch -- an ABI or CUDA-runtime "
                      "mismatch. This is the state where `import xformers` still succeeds and "
                      "every attention_backend=xformers run dies in the forward.")
            except Exception as e:  # noqa: BLE001
                check("xformers C++/CUDA extension status readable", False, f"{type(e).__name__}: {e}")

            # Op-level truth: a build can load and still be missing the kernel a run needs.
            # Require at least one AVAILABLE forward (F) and one backward (B) among the
            # memory_efficient_attention ops -- half-built wheels routinely have the forward and
            # not the backward, which passes training setup and dies at the first loss.backward().
            # Suffixes are NOT quality marks: `-pt` means the op is reached through PyTorch's own
            # bindings (e.g. `fa2F@2.5.7-pt` is genuine FlashAttention-2), and current xformers
            # deliberately routes there instead of shipping duplicate kernels. Unavailable `ck*`
            # (ROCm) and `fa3*` (Hopper-only) entries on an NVIDIA Ampere box are expected.
            try:
                from xformers.info import get_features_status
                mea = {k.split(".", 1)[1]: v for k, v in get_features_status().items()
                       if k.startswith("memory_efficient_attention.")}
                avail = [k for k, v in mea.items() if v == "available"]
                fwd = [k for k in avail if k.split("@")[0].removesuffix("-pt").endswith("F")]
                bwd = [k for k in avail if k.split("@")[0].removesuffix("-pt").endswith("B")]
                check("xformers memory_efficient_attention has a forward AND a backward kernel",
                      bool(fwd) and bool(bwd),
                      f"forward={sorted(fwd) or 'none'}; backward={sorted(bwd) or 'none'} "
                      f"({len(avail)}/{len(mea)} ops available)",
                      "a forward-only build passes training setup and the first forward, then "
                      "dies at the first loss.backward(). Rebuild against this torch, or use "
                      "attention_backend=native|flex|flash.")
            except Exception as e:  # noqa: BLE001
                check("xformers op dispatcher readable", False, f"{type(e).__name__}: {e}")

            if args.gpu and torch.cuda.is_available():
                from xformers.ops import fmha
                q = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.float16,
                                requires_grad=True)
                bias = fmha.BlockDiagonalMask.from_seqlens([10, 6])
                out = fmha.memory_efficient_attention(q, q, q, attn_bias=bias)
                check("memory_efficient_attention runs (BlockDiagonal, fp16)",
                      bool(out.isfinite().all().item()))
                # The backward is a separate kernel and a separate way to be broken.
                try:
                    out.square().sum().backward()
                    check("memory_efficient_attention backward runs",
                          bool(q.grad is not None and q.grad.isfinite().all().item()))
                except Exception as e:  # noqa: BLE001
                    check("memory_efficient_attention backward runs", False,
                          f"{type(e).__name__}: {e}",
                          "forward-only build: inference and the first training step's forward "
                          "both pass, then the first backward dies")
        except Exception as e:  # noqa: BLE001 -- certification must report, not crash
            check("xformers usable", False, f"{type(e).__name__}: {e}",
                  "see the xformers section of docs/OSCAR.md (tag choice, flash decision tree)")

    # ---- 4b. the check that actually predicts a crash -------------------------------
    # A working `import xformers` does NOT mean a run can use it: lgatr and lloca each keep
    # a backend REGISTRY populated at import time, and an ABI-mismatched xformers (built for
    # a different torch) is silently omitted from both. A model with
    # `attention_backend: xformers` then dies on its first forward -- past init, minutes or
    # hours into a job. This asserts the registries, not the import.
    pinned = "xformers"
    for lib, probe in (("lgatr", _lgatr_backends), ("lloca", _lloca_backends)):
        try:
            registry = probe()
        except Exception as e:  # noqa: BLE001
            # NEVER swallow this into "the backend is missing". The previous version read
            # `getattr(mod, "_REGISTRY", {})`, so when lgatr 2.0 removed its eager registry the
            # default turned an API change into `registry=[]` and a confident FAIL saying
            # xformers was unbound -- on an environment where it binds fine. "I could not ask
            # the question" and "the answer is no" must never render as the same result.
            check(f"{lib} backend availability is introspectable", False,
                  f"{type(e).__name__}: {e}",
                  f"{lib}'s backend API changed shape again -- env_check cannot tell whether "
                  f"'{pinned}' is bound, so treat this as UNKNOWN, not as a working env. Fix "
                  f"the probe in env_check.py's section 4b before trusting any run.")
            continue
        available = pinned in registry
        if xformers is None and not available:
            print(f"[INFO] {lib} backends: {registry} (no xformers, as expected)")
            continue
        if not cuda_here and not available:
            # BOTH libraries gate the xformers (and flash) backends on CUDA being visible, so
            # off-GPU the registry is ['flex', 'native'] for a perfectly healthy install. This
            # check is UNEVALUATABLE here, not failed -- reporting FAIL on a login node is how
            # an env check teaches you to ignore its FAILs, and a real ABI mismatch looks
            # identical. Re-run on a GPU allocation to actually test it.
            print(f"[SKIP] {lib} '{pinned}' backend: registry={registry} "
                  f"-- CUDA-gated, unevaluatable off-GPU; re-run inside `interact -g 1`")
            continue
        check(f"{lib} can bind the '{pinned}' backend", available,
              f"registry={registry}",
              "xformers imports but neither library could bind it -- almost always an ABI "
              "mismatch with the installed torch. Any config pinning attention_backend="
              f"{pinned} will crash IN THE FORWARD, not at init. Rebuild xformers against "
              "this torch, or override the backend to native|flex.")

    # ---- 5. flash-attn (informational; PT-flash floor exists regardless) ----------
    try:
        import flash_attn
        print(f"[INFO] container flash_attn imports: {flash_attn.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"[INFO] flash_attn not importable ({type(e).__name__}) -- fine: xformers "
              "falls back to bundled/PT-flash; lgatr's registry skips the flash backend")

    # ---- summary ------------------------------------------------------------------
    fails = RESULTS.count(False)
    print(f"\n{'CERTIFIED' if fails == 0 else 'NOT CERTIFIED'}: "
          f"{RESULTS.count(True)}/{len(RESULTS)} checks passed"
          + ("" if args.gpu else
             "  (a GPU IS visible here -- re-run with --gpu to certify the kernels)"
             if cuda_here else
             "  (CPU-context run; repeat with --gpu on a GPU allocation before a campaign)"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
