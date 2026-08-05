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
    if xformers is not None:
        try:
            ver = xformers.__version__
            check("xformers real build (sha-stamped)", "+" in ver, ver,
                  "a bare version = the crippled kernel-free sdist build; rebuild from the "
                  "git tag matched to the container torch (see OSCAR.md)")
            import xformers.ops as xops  # the import that the flash chain can crash
            check("xformers.ops imports (flash chain resolved)", True)
            if args.gpu and torch.cuda.is_available():
                from xformers.ops import fmha
                q = torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.float16)
                bias = fmha.BlockDiagonalMask.from_seqlens([10, 6])
                out = fmha.memory_efficient_attention(q.view(1, 16, 4, 32), q.view(1, 16, 4, 32),
                                                      q.view(1, 16, 4, 32), attn_bias=bias)
                check("memory_efficient_attention runs (BlockDiagonal, fp16)",
                      bool(out.isfinite().all().item()))
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
    for mod_name in ("lgatr.primitives.attention_backends", "lloca.backbone.attention_backends.mask"):
        try:
            registry = sorted(getattr(importlib.import_module(mod_name), "_REGISTRY", {}))
        except Exception as e:  # noqa: BLE001
            check(f"{mod_name} importable", False, f"{type(e).__name__}: {e}")
            continue
        available = pinned in registry
        if xformers is None and not available:
            print(f"[INFO] {mod_name.split('.')[0]} backends: {registry} (no xformers, as expected)")
            continue
        check(f"{mod_name.split('.')[0]} registered the '{pinned}' backend", available,
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
          + ("" if args.gpu else "  (CPU-context run; repeat with --gpu before a campaign)"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
