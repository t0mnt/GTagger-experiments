"""Phase 0 golden fixtures for the lgatr 1.4.4 -> 2.0.0 migration (docs/lgatr2-migration.md).

Record mode (ONLY on lgatr 1.4.4 -- hard-asserted):
    LGATR_PARITY=record python -m pytest tests/experiments/test_lgatr_migration_parity.py -q
writes, under tests/fixtures/lgatr144/:
  - production_manifests.json : per full-size config, the total parameter count and the sorted
    multiset of "(shape)|requires_grad" entries (Gate B compares THESE, never key names --
    Phase -1 exhibited two models with equal totals and different tensors, so the multiset,
    not the total, is the check), plus key names as KEY_MAP raw material.
  - <model>.pt : reduced-config transplant packs from the config_quick tree -- qkv-bias-
    normalized (S5) fp64 state_dict, the fixed input batches (a real mini-dataset batch and an
    edge-case truncation with multiplicities [1, knn_k-1, mid, full]), final outputs, per-block
    activations, gradient pack (d loss / d x plus per-parameter grad norms), and the resolved
    config snapshot.

Check mode (default, no env var):
  - on lgatr 1.4.x with fixtures present: SELF-CONSISTENCY -- rebuild each reduced model,
    load the recorded state_dict, and require bit-identical outputs/activations (proves the
    harness round-trips before any cross-version claim is made).
  - on lgatr 2.0.x: the transplant path (KEY_MAP remap + S6 sqrt(2) GLU-gate compensation);
    SKIPS until tests/fixtures/lgatr144/key_map.json exists (built in Task B, Phase 1a).
  - without fixtures: skips cleanly.

Determinism contract: the CANONICAL record invocation is the orchestrator
    LGATR_PARITY=record python tests/experiments/test_lgatr_migration_parity.py
which records every model in its own pristine subprocess and writes content_hashes.json;
run it twice and content_hashes.json must be byte-identical. Two measured facts shape this
contract (neither is hypothesis -- both were isolated empirically during Phase 0):

1. Per-model subprocess isolation is load-bearing: executing FlexAttention (the equivectors
   composition's only CPU-capable backend) perturbs global torch state so that every
   SUBSEQUENT forward in the same process shifts by ~1 ULP (~1e-16 relative). Pristine
   single-model processes are exactly reproducible; full-pytest-session records are
   internally consistent but differ from pristine ones on every model that runs after
   equivectors. The in-pytest self-consistency check therefore asserts deviation <= 1e-12
   rather than bit-equality (the discrimination argument is untouched: real porting
   mistakes land at O(1e-2..1), twelve orders above).
2. The contract hashes CONTENT, not file bytes: torch.save's zip layout (storage-key
   assignment) is process-dependent, so two .pt files with field-by-field torch.equal
   content routinely differ in sha256. content_hashes.json holds the canonical hash
   (sorted-key walk, tensor dtype+shape+bytes); test_fixture_content_hashes re-derives it
   from the .pt files, so corruption or silent regeneration is caught in every check run.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "lgatr144"
RECORD = os.environ.get("LGATR_PARITY") == "record"

# Byte-stable fixtures require run-context-independent arithmetic: multi-threaded CPU reductions
# reorder at the ULP level depending on the surrounding process (observed: ~1e-16 relative on
# outputs between full-suite and single-test invocations). One thread makes the tiny fixture
# models bitwise reproducible everywhere.
torch.set_num_threads(1)

# S5: biases living under a q/k/v projection -- slim fused `attention.linear_in`, full-LGATr
# `attention.qkv_module` (observed names: ...attention.linear_in.linear_s.bias,
# ...attention.qkv_module.in_linear.{s2mvs,mvs2s}.bias). out_linear biases are NOT qkv.
QKV_BIAS_RE = re.compile(r"\.attention\.(qkv_module|linear_in)\..*bias$")

# Reduced-config (config_quick) fixture set. hidden_v_channels is forced OFF the value 4
# wherever slim blocks appear: 4 is the one width where a missed M8 channel-last transpose
# becomes shape-legal and silent (runbook H13).
REDUCED = {
    "tag_lgatr": ["model=tag_lgatr"],
    "tag_slim": ["model=tag_slim", "model.net.hidden_v_channels=6"],
    "tag_CGENNLGATrGraphTrans": ["model=tag_CGENNLGATrGraphTrans"],
    "tag_CGENNLGATrGraphGPS": ["model=tag_CGENNLGATrGraphGPS"],
    "tag_LorentzNetLGATrSlimGraphTrans": [
        "model=tag_LorentzNetLGATrSlimGraphTrans", "model.net.hidden_v_channels=6"],
    "tag_LorentzNetLGATrSlimGraphGPS": [
        "model=tag_LorentzNetLGATrSlimGraphGPS", "model.net.hidden_v_channels=6"],
    # LGATrVectors insists on a SPARSE lloca attention backend; on CPU the only importable one
    # is flex, whose backward is unsupported on CPU -> this composition records WITHOUT the
    # gradient pack (reason stored in the fixture; frames-path backward coverage moves to
    # Gate G on cluster). Deliberate documented narrowing, not a model dropped.
    "equivectors_lgatr": [
        "model=tag_PlainGraphTrans", "model/framesnet=learnedso13",
        "model/framesnet/equivectors=lgatr",
        "+model.framesnet.equivectors.attention_backend=flex"],
}
NO_GRAD_PACK = {
    "equivectors_lgatr": "FlexAttention has no CPU backward; xformers/flash are CPU-gated "
                         "by lloca's registry and varlen is CUDA-only",
}
# The pure-baseline wrappers in-place-modify a view of the spurion-augmented input assembly;
# with data.x as a grad leaf that view enters the autograd graph and backward raises
# ("modified by an inplace operation"). Training never differentiates w.r.t. inputs, so this
# is invisible upstream; here it means: record param-grad norms (x not a leaf), skip x_grad.
PARAMS_ONLY_GRAD = {
    "tag_slim": "wrapper in-place op on the input assembly blocks input-leaf backward",
    "tag_lgatr": "wrapper in-place op on the input assembly blocks input-leaf backward",
}
# Production manifests use the SAME override lists against the full config tree.
PRODUCTION = REDUCED


def _lgatr_version():
    import lgatr
    return lgatr.__version__


def _build(config_dir, overrides, with_data):
    import logging.handlers  # noqa: F401  (experiments.logger assumes it is imported)
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    base = ["save=false", "use_float64=true", "training.batchsize=4", "data.dataset=mini"]
    with hydra.initialize_config_dir(config_dir=str(REPO / config_dir), version_base=None):
        cfg = hydra.compose(config_name="toptagging", overrides=base + list(overrides))
    torch.manual_seed(0)
    exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    if with_data:
        exp.init_data()
        exp._init_dataloader()
    exp._init_loss()  # _get_ypred_and_label needs exp.loss even without data
    exp.model.eval()
    return exp


def _first_batch(exp):
    torch.manual_seed(1)
    return next(iter(exp.train_loader))


def _truncate_jets(data, keep_counts):
    """Edge-case batch: keep the first n_i particles of jet i, rebuild batch/ptr."""
    ptr = data.ptr.tolist()
    keep_rows = []
    for j, n in enumerate(keep_counts):
        n = min(n, ptr[j + 1] - ptr[j])
        keep_rows.extend(range(ptr[j], ptr[j] + n))
    idx = torch.tensor(keep_rows, dtype=torch.long)
    out = data.clone()
    n_rows = data.x.shape[0]
    for key in data.keys():
        val = data[key]
        if torch.is_tensor(val) and val.dim() >= 1 and val.shape[0] == n_rows:
            out[key] = val.index_select(0, idx)
    counts = [min(n, ptr[j + 1] - ptr[j]) for j, n in enumerate(keep_counts)]
    out.ptr = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.long)
    out.batch = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    return out


def _batch_fields(data):
    return {k: data[k].clone() for k in data.keys() if torch.is_tensor(data[k])}


def _rebuild_batch(fields):
    from torch_geometric.data import Data
    d = Data()
    for k, v in fields.items():
        d[k] = v.clone()
    return d


def _grad_mode(name):
    if name in NO_GRAD_PACK:
        return "none"
    if name in PARAMS_ONLY_GRAD:
        return "params_only"
    return "full"


def _zero_qkv_biases(model):
    zeroed = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if QKV_BIAS_RE.search(name):
                p.zero_()
                zeroed.append(name)
    return zeroed


def _block_hook_targets(model):
    """Ordered transformer-ish stages: the i-th child of any ModuleList named blocks/layers."""
    pat = re.compile(r"\.(blocks|layers)\.\d+$")
    return [(n, m) for n, m in model.named_modules() if pat.search("." + n)]


def _forward_pack(exp, data, with_grads=True):
    """One instrumented forward(+backward): outputs, per-block activations, gradient pack."""
    acts, handles = [], []
    for name, mod in _block_hook_targets(exp.model):
        def hook(mod, inp, out, name=name):
            tensors = [t.detach().clone() for t in
                       (out if isinstance(out, (tuple, list)) else (out,)) if torch.is_tensor(t)]
            acts.append((name, tensors))
        handles.append(mod.register_forward_hook(hook))
    try:
        work = data.clone()
        if with_grads == "full":
            work.x = work.x.detach().clone().requires_grad_(True)
            leaf = work.x
        if with_grads in ("full", "params_only"):
            y = exp._get_ypred_and_label(work)[0]
            loss = y.square().sum()
            loss.backward()
            # zero-signal params (dead vector tails when out_v_channels=0, e.g. slim
            # mlp/linear_out weight_v) flip between grad=None and grad=0.0 depending on
            # autograd's internal scheduling -- both mean "no gradient signal"; normalize to
            # None so the fixture bytes do not depend on which one autograd produced.
            def gnorm(p):
                if p.grad is None:
                    return None
                n = float(p.grad.norm().item())
                return None if n == 0.0 else n
            grad_norms = [[n, gnorm(p)] for n, p in exp.model.named_parameters()]
            pack = {
                "y": y.detach().clone(),
                "x_grad": leaf.grad.detach().clone() if with_grads == "full" else None,
                "param_grad_norms": grad_norms,
                "block_acts": acts,
            }
            exp.model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                y = exp._get_ypred_and_label(work)[0]
            pack = {"y": y.detach().clone(), "x_grad": None,
                    "param_grad_norms": None, "block_acts": acts}
        return pack
    finally:
        for h in handles:
            h.remove()


def _manifest(model):
    entries = sorted(f"{tuple(p.shape)}|rg={p.requires_grad}" for p in model.parameters())
    return {
        "total": int(sum(p.numel() for p in model.parameters())),
        "entries": entries,
        "names": [[n, list(p.shape), bool(p.requires_grad)]
                  for n, p in model.named_parameters()],
    }


def _record_pack(name, overrides):
    exp = _build("config_quick", overrides, with_data=True)
    zeroed = _zero_qkv_biases(exp.model)
    from omegaconf import OmegaConf
    data_main = _first_batch(exp)
    counts = [data_main.ptr[i + 1] - data_main.ptr[i] for i in range(len(data_main.ptr) - 1)]
    knn_k = 4  # config_quick hybrids use knn_k/k = 4; keep one jet strictly below it
    keep = [1, knn_k - 1, max(2, int(counts[2]) // 2), int(counts[3])]
    data_edge = _truncate_jets(data_main, keep)
    mode = _grad_mode(name)
    # cfg snapshot: strip run-bookkeeping lines -- run_dir carries a fresh random suffix every
    # invocation, which would defeat the double-record byte-identity contract while carrying
    # zero model information. Everything model-relevant stays.
    volatile = ("run_dir:", "run_name:", "run_idx:", "db:", "artifacts:")
    cfg_yaml = "\n".join(l for l in OmegaConf.to_yaml(exp.cfg, resolve=True).splitlines()
                         if not l.strip().startswith(volatile))
    pack = {
        "lgatr_version": _lgatr_version(),
        "sd": exp.model.state_dict(),
        "zeroed_qkv_biases": zeroed,
        "grad_mode": mode,
        "grad_limit_reason": NO_GRAD_PACK.get(name) or PARAMS_ONLY_GRAD.get(name),
        "cfg_yaml": cfg_yaml,
        "batch_main": _batch_fields(data_main),
        "batch_edge": _batch_fields(data_edge),
        "main": _forward_pack(exp, data_main, mode),
        "edge": _forward_pack(exp, data_edge, mode),
    }
    FIX.mkdir(parents=True, exist_ok=True)
    torch.save(pack, FIX / f"{name}.pt")


def _self_consistency(name, overrides):
    ref = torch.load(FIX / f"{name}.pt", weights_only=False)
    exp = _build("config_quick", overrides, with_data=False)
    exp.model.load_state_dict(ref["sd"], strict=True)
    mode = _grad_mode(name)
    # <= 1e-12, not bit-equality: cross-test global-state ULP (see module docstring). Real
    # mistakes are O(1e-2..1); anything in between is a finding, not a rounding artifact.
    ulp = 1e-12
    def dev(a, b):
        return (a - b).abs().max().item()
    for tag in ("main", "edge"):
        data = _rebuild_batch(ref[f"batch_{tag}"])
        pack = _forward_pack(exp, data, mode)
        assert dev(pack["y"], ref[tag]["y"]) <= ulp, (
            f"{name}/{tag}: outputs deviate {dev(pack['y'], ref[tag]['y']):.3e}")
        if mode == "full":
            assert dev(pack["x_grad"], ref[tag]["x_grad"]) <= ulp, f"{name}/{tag}: input-grad drifted"
        if mode in ("full", "params_only"):
            for (n1, g1), (n2, g2) in zip(pack["param_grad_norms"], ref[tag]["param_grad_norms"]):
                same = (g1 is None) == (g2 is None) and (
                    g1 is None or abs(g1 - g2) <= 1e-9 * (1 + abs(g2)))
                assert same, f"{name}/{tag}: grad norm drift at {n1}: {g1} vs {g2}"
        assert len(pack["block_acts"]) == len(ref[tag]["block_acts"]), f"{name}/{tag}: block count"
        for (n1, ts1), (n2, ts2) in zip(pack["block_acts"], ref[tag]["block_acts"]):
            for a, b in zip(ts1, ts2):
                assert dev(a, b) <= ulp, f"{name}/{tag}: first divergence at block {n1}"


def _transplant_check(name, overrides):  # Task B fills key_map.json; the machinery is ready
    key_map_file = FIX / "key_map.json"
    if not key_map_file.exists():
        pytest.skip("KEY_MAP not built yet (migration Task B, Phase 1a)")
    key_map = json.loads(key_map_file.read_text())
    ref = torch.load(FIX / f"{name}.pt", weights_only=False)
    sd = {key_map.get(name, {}).get(k, k): v for k, v in ref["sd"].items()}
    sd = _rescale_glu_gates(sd, key_map.get(f"{name}::glu_weight_v_keys", []))
    exp = _build("config_quick", overrides, with_data=False)
    missing, unexpected = exp.model.load_state_dict(sd, strict=False)
    waived = set(key_map.get(f"{name}::waived", []))
    assert set(missing) | set(unexpected) <= waived, (
        f"{name}: unwaived state_dict mismatch: missing={missing} unexpected={unexpected}")
    tier1 = os.environ.get("LGATR_PARITY_TIER", "1") == "1"
    tol = 1e-10 if tier1 else 1e-8
    mode = _grad_mode(name)
    for tag in ("main", "edge"):
        data = _rebuild_batch(ref[f"batch_{tag}"])
        pack = _forward_pack(exp, data, mode)
        rel = (pack["y"] - ref[tag]["y"]).abs().max() / (1 + ref[tag]["y"].abs().max())
        assert rel < tol, f"{name}/{tag}: forward parity {rel:.3e} >= {tol}"
        if mode == "full":
            relg = ((pack["x_grad"] - ref[tag]["x_grad"]).abs().max()
                    / (1 + ref[tag]["x_grad"].abs().max()))
            assert relg < (tol if tier1 else 1e-6), f"{name}/{tag}: grad parity {relg:.3e}"


def _rescale_glu_gates(sd, glu_weight_v_keys):
    """S6 compensation: x sqrt(2) on both vector-gate chunks of each GLU fused linear."""
    out = dict(sd)
    for key in glu_weight_v_keys:
        w = out[key].clone()
        n = w.shape[0] // 3
        w[n:] = w[n:] * (2.0 ** 0.5)
        out[key] = w
    return out


@pytest.mark.parametrize("name", sorted(PRODUCTION))
def test_production_manifest(name):
    manifest_file = FIX / "production_manifests.json"
    if RECORD:
        assert _lgatr_version() == "1.4.4", "record mode must run on lgatr 1.4.4"
        exp = _build("config", PRODUCTION[name], with_data=False)
        FIX.mkdir(parents=True, exist_ok=True)
        stored = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        stored[name] = _manifest(exp.model)
        manifest_file.write_text(json.dumps(stored, indent=1, sort_keys=True))
        return
    if not manifest_file.exists():
        pytest.skip("no 1.4.4 fixtures recorded")
    if _lgatr_version().startswith("2."):
        pytest.skip("Gate B on 2.x applies rule-derived waivers (migration Task B)")
    stored = json.loads(manifest_file.read_text())[name]
    exp = _build("config", PRODUCTION[name], with_data=False)
    live = _manifest(exp.model)
    assert live["total"] == stored["total"] and live["entries"] == stored["entries"], (
        f"{name}: production manifest drifted from recorded 1.4.4 baseline")


@pytest.mark.parametrize("name", sorted(REDUCED))
def test_transplant_parity(name):
    if RECORD:
        assert _lgatr_version() == "1.4.4", "record mode must run on lgatr 1.4.4"
        _record_pack(name, REDUCED[name])
        return
    if not (FIX / f"{name}.pt").exists():
        pytest.skip("no 1.4.4 fixtures recorded")
    if _lgatr_version().startswith("1.4"):
        _self_consistency(name, REDUCED[name])
    else:
        _transplant_check(name, REDUCED[name])


def _content_hash(obj):
    """Canonical content hash: independent of torch.save's process-dependent zip layout."""
    h = hashlib.sha256()

    def feed(x, path):
        h.update(path.encode())
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                feed(x[k], f"{path}/{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                feed(v, f"{path}[{i}]")
        elif torch.is_tensor(x):
            h.update(str(x.dtype).encode())
            h.update(str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(x).encode())

    feed(obj, "")
    return h.hexdigest()


def test_fixture_content_hashes():
    """Every fixture's canonical content hash must match content_hashes.json (recorded)."""
    hash_file = FIX / "content_hashes.json"
    if not hash_file.exists():
        pytest.skip("no fixtures recorded")
    stored = json.loads(hash_file.read_text())
    for fname, expected in sorted(stored.items()):
        path = FIX / fname
        assert path.exists(), f"fixture {fname} missing"
        if fname.endswith(".pt"):
            live = _content_hash(torch.load(path, weights_only=False))
        else:
            live = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{live}  {fname}")
        assert live == expected, f"{fname}: content hash mismatch (fixture corrupted or regenerated)"


if __name__ == "__main__":
    # Canonical record orchestrator: one pristine subprocess per model (see docstring for why).
    import subprocess
    import sys

    assert os.environ.get("LGATR_PARITY") == "record", (
        "run with LGATR_PARITY=record; check mode runs via pytest")
    me = str(Path(__file__).resolve())
    for name in sorted(REDUCED):
        for test in (f"test_production_manifest[{name}]", f"test_transplant_parity[{name}]"):
            r = subprocess.run([sys.executable, "-m", "pytest", f"{me}::{test}", "-q"],
                               cwd=REPO, capture_output=True, text=True, timeout=1200)
            status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
            print(f"{name:<36} {test.split('[')[0]:<26} {status}")
            if r.returncode != 0:
                sys.exit(f"record failed for {test}")
    hashes = {}
    for f in sorted(FIX.iterdir()):
        if f.name == "content_hashes.json":
            continue
        if f.name.endswith(".pt"):
            hashes[f.name] = _content_hash(torch.load(f, weights_only=False))
        else:
            hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (FIX / "content_hashes.json").write_text(json.dumps(hashes, indent=1, sort_keys=True))
    print("\ncanonical content hashes (the double-record comparison target):")
    for k, v in hashes.items():
        print(f"{v}  {k}")
