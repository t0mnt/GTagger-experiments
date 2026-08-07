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


# Phase 3 parity pins: applied ONLY to the parity script's v2 builds (shipped configs are
# v2-native, Posture B). S1 -> nonlinearity_v=null restores the v1 gate routing; S2 ->
# norm_elementwise_affine=false removes the gains the pinned build must not have.
PARITY_PINS = {
    "tag_slim": ["+model.net.nonlinearity_v=null", "+model.net.norm_elementwise_affine=false"],
    "tag_lgatr": ["+model.net.norm_elementwise_affine=false"],
    "tag_LorentzNetLGATrSlimGraphTrans": [
        "+model.net.nonlinearity_v=null", "+model.net.norm_elementwise_affine=false"],
    "tag_LorentzNetLGATrSlimGraphGPS": ["+model.net.nonlinearity_v=null"],
    "tag_CGENNLGATrGraphTrans": ["+model.net.norm_elementwise_affine=false"],
    "tag_CGENNLGATrGraphGPS": [],
    "equivectors_lgatr": ["+model.framesnet.equivectors.net.norm_elementwise_affine=false"],
}
# Tier 1 additionally forces the DENSE geometric product on full-LGATr-bearing models (S3);
# slim models have no geometric product.
TIER1_SPARSE_GP_OFF = {
    "tag_lgatr": ["+model.net.primitives.sparse_gp=false"],
    "tag_CGENNLGATrGraphTrans": ["+model.net.primitives.sparse_gp=false"],
    "tag_CGENNLGATrGraphGPS": ["+model.net.primitives_sparse_gp=false"],
    "equivectors_lgatr": ["+model.framesnet.equivectors.net.primitives.sparse_gp=false"],
}
# S6 compensation targets, derived BY RULE from the v1 state_dict: every slim GLU fused
# linear's vector weight. S5 waivers likewise: every bias under a qkv projection module.
GLU_WEIGHT_V_RE = re.compile(r"\.mlp\.layers\.\d+\.linear\.weight_v$")


def _transplant_check(name, overrides):
    ref = torch.load(FIX / f"{name}.pt", weights_only=False)
    key_map_file = FIX / "key_map.json"
    key_map = json.loads(key_map_file.read_text()) if key_map_file.exists() else {}
    sd = {key_map.get(name, {}).get(k, k): v for k, v in ref["sd"].items()}
    glu_keys = [k for k in sd if GLU_WEIGHT_V_RE.search(k)]
    sd = _rescale_glu_gates(sd, glu_keys)
    tier1 = os.environ.get("LGATR_PARITY_TIER", "1") == "1"
    pins = PARITY_PINS[name] + (TIER1_SPARSE_GP_OFF.get(name, []) if tier1 else [])
    if True:  # S9 compensation applies in BOTH tiers: it bridges the v1<->v2 gelu-flavor
        # change so that tier 2 isolates exactly one reorder (sparse_gp) at its 1e-8 bar,
        # instead of measuring the shipped S9 delta (~1e-4) and failing by construction.
        # S9 compensation (verification instrument only): v2 unified get_nonlinearity uses
        # gelu(approximate="tanh"); v1 used exact erf-GeLU. Restore erf for the pinned build
        # so tier 1 isolates everything else at 1e-10; shipped models keep v2's tanh-gelu.
        # v1's flavors were SPLIT: slim used exact erf-GeLU everywhere; the full model's
        # ScalarGatedNonlinearity used tanh-GeLU for the multivector GATE (v1 gated_gelu,
        # approximate="tanh") but exact erf-GeLU on the auxiliary scalar stream. Reproduce
        # exactly that during the pinned tier-1 build; shipped models keep v2's unified tanh.
        import torch.nn.functional as F
        import lgatr.layers.mlp.nonlinearities as _c1
        import lgatr.layers.slim_layers as _c2
        import lgatr.utils.misc as _lm
        _orig_get = _c2.get_nonlinearity
        _c2.get_nonlinearity = lambda label: (
            (lambda x: F.gelu(x)) if label == "gelu" else _orig_get(label))
        _orig_fwd = _c1.ScalarGatedNonlinearity.forward
        def _v1_split_forward(self, multivectors, scalars=None):
            weights = F.gelu(multivectors[..., [0]], approximate="tanh")
            outputs_mv = weights * multivectors
            outputs_s = F.gelu(scalars) if scalars is not None else None
            return outputs_mv, outputs_s
        _c1.ScalarGatedNonlinearity.forward = _v1_split_forward
        # NB: slim's flavor binds at CONSTRUCTION (get_nonlinearity), the full model's
        # ScalarGated flavor at FORWARD time -> the patch must span the comparisons below,
        # not just the build; it is restored at the end of this check.
    exp = _build("config_quick", list(overrides) + pins, with_data=False)
    missing, unexpected = exp.model.load_state_dict(sd, strict=False)
    # v1-side keys with no v2 slot, derived by rule: S5 qkv biases + the attention `metric`
    # buffer v2 made non-persistent (constant Minkowski signature, zero learnable content).
    expected_extra = {k for k in sd
                      if QKV_BIAS_RE.search(k) or k.endswith(".attention.metric")}
    assert set(unexpected) == expected_extra and not missing, (
        f"{name}: state_dict mismatch beyond the S5/metric rules: "
        f"missing={sorted(missing)} unexpected={sorted(set(unexpected) - expected_extra)}")
    tier1 = os.environ.get("LGATR_PARITY_TIER", "1") == "1"
    tol = 1e-10 if tier1 else 1e-8
    mode = _grad_mode(name)
    learned_frames = any("framesnet=learned" in o for o in overrides)
    for tag in ("main", "edge"):
        data = _rebuild_batch(ref[f"batch_{tag}"])
        # S10 (operator ruling 2026-08-07): a multiplicity-1 jet degenerates the LEARNED-frames
        # path, and its conditioning amplifies the fp64 cross-version reassociation baseline
        # (decision-log evidence: the n=1 jet alone at 2.8e-9 tier-1 / 6.3e-6 tier-2 at the
        # output, peaking at 6.9e-4 inside the frames net's block 0; every n>=2 jet at
        # <= 4.6e-14). No campaign jet has multiplicity 1, so degenerate-frame inputs are out
        # of scope for the strict bar. The rule is derived, not listed: it fires only for a
        # learned-frames composition on a batch actually containing an n=1 jet.
        degenerate = learned_frames and int(data.ptr.diff().min()) == 1
        pack = _forward_pack(exp, data, mode)
        if degenerate:
            trip = 1e-6 if tier1 else 1e-4
            keep = data.ptr.diff() >= 2
            y, yr = pack["y"], ref[tag]["y"]
            rel_keep = (y[keep] - yr[keep]).abs().max() / (1 + yr[keep].abs().max())
            rel_deg = (y[~keep] - yr[~keep]).abs().max() / (1 + yr[~keep].abs().max())
            print(f"GATE-C {name}/{tag} tier{1 if tier1 else 2} forward rel(n>=2)="
                  f"{rel_keep:.3e} rel(n=1)={rel_deg:.3e} (S10 tripwire {trip:.0e})")
            assert rel_keep < tol, (f"{name}/{tag}: n>=2 jets must meet the strict bar even "
                                    f"in the degenerate batch: {rel_keep:.3e} >= {tol}")
            # the tripwire sits 2-4 orders below mistake scale O(1e-2..1): frames-path
            # breakage still fails loudly, here AND via the strict n>=2 assert above.
            assert rel_deg < trip, (f"{name}/{tag}: degenerate n=1 jet beyond the S10 "
                                    f"tripwire: {rel_deg:.3e} >= {trip}")
            # Per-block activations of the degenerate batch are diagnostic, not gate-bearing:
            # the n=1 jet's particle rows thread every block tensor, and masking jets inside
            # arbitrary per-block shapes is exactly where silent-alias mistakes live (H13).
            # Full per-block strictness is retained on the main batch. (continue also skips
            # the input-grad check -- no learned-frames fixture records one; flex has no CPU
            # backward.)
            continue
        rel = (pack["y"] - ref[tag]["y"]).abs().max() / (1 + ref[tag]["y"].abs().max())
        print(f"GATE-C {name}/{tag} tier{1 if tier1 else 2} forward rel={rel:.3e}")
        assert rel < tol, f"{name}/{tag}: forward parity {rel:.3e} >= {tol}"
        for i, ((n1, ts1), (n2, ts2)) in enumerate(zip(pack["block_acts"], ref[tag]["block_acts"])):
            for a, b in zip(ts1, ts2):
                if a.shape != b.shape and a.shape[-2:] == b.shape[-2:][::-1]:
                    a = a.transpose(-1, -2)  # M8: v2 slim blocks emit channel-last (H14(2))
                d = (a - b).abs().max() / (1 + b.abs().max())
                assert d < tol, f"{name}/{tag}: first divergence at block {n1} (rel={d:.3e})"
        if mode == "full":
            relg = ((pack["x_grad"] - ref[tag]["x_grad"]).abs().max()
                    / (1 + ref[tag]["x_grad"].abs().max()))
            print(f"GATE-C {name}/{tag} grad rel={relg:.3e}")
            assert relg < (tol if tier1 else 1e-6), f"{name}/{tag}: grad parity {relg:.3e}"
    _c2.get_nonlinearity = _orig_get
    _c1.ScalarGatedNonlinearity.forward = _orig_fwd


def _rescale_glu_gates(sd, glu_weight_v_keys):
    """S6 compensation: x sqrt(2) on both vector-gate chunks of each GLU fused linear."""
    out = dict(sd)
    for key in glu_weight_v_keys:
        w = out[key].clone()
        n = w.shape[0] // 3
        w[n:] = w[n:] * (2.0 ** 0.5)
        out[key] = w
    return out


REBASELINE = os.environ.get("LGATR_PARITY") == "rebaseline"
V2_MANIFESTS = FIX / "production_manifests_v2.json"


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
    stored = json.loads(manifest_file.read_text())[name]
    if REBASELINE:
        # Posture-flip re-baseline (Task C step 1): record the v2-DEFAULT manifests -- but
        # only after rule-checking them against v1 (never trust a recording you didn't check).
        assert _lgatr_version().startswith("2."), "re-baseline records v2 defaults on 2.x"
        exp = _build("config", PRODUCTION[name], with_data=False)  # unpinned: shipped config
        live = _manifest(exp.model)
        _rebaseline_rule_check(name, stored, exp, live)
        v2_stored = json.loads(V2_MANIFESTS.read_text()) if V2_MANIFESTS.exists() else {}
        v2_stored[name] = live
        V2_MANIFESTS.write_text(json.dumps(v2_stored, indent=1, sort_keys=True))
        return
    if _lgatr_version().startswith("2."):
        _manifest_check_v2(name, stored)
        if V2_MANIFESTS.exists():
            # Post-flip regression gate: the shipped (unpinned, v2-default) build must match
            # the re-baselined manifest exactly -- names, shapes, requires_grad, total.
            baseline = json.loads(V2_MANIFESTS.read_text())[name]
            live = _manifest(_build("config", PRODUCTION[name], with_data=False).model)
            assert (live["total"] == baseline["total"]
                    and live["entries"] == baseline["entries"]
                    and live["names"] == baseline["names"]), (
                f"{name}: shipped-config manifest drifted from the Posture-B v2 baseline")
        return
    exp = _build("config", PRODUCTION[name], with_data=False)
    live = _manifest(exp.model)
    assert live["total"] == stored["total"] and live["entries"] == stored["entries"], (
        f"{name}: production manifest drifted from recorded 1.4.4 baseline")


def _rebaseline_rule_check(name, stored_v1, exp, live):
    """The Posture-B re-baseline is itself rule-checked (runbook Phase 3 / Gate B row):
    v2-default manifest == v1 manifest − S5 qkv biases + affine norm gains, where the gain
    set is collected STRUCTURALLY from the live model (norm-owned parameters only, same
    helper the H15 optimizer exemption uses) and requires_grad must follow the freezing rule
    on old and new parameters alike.
    """
    from math import prod
    from experiments.base_experiment import lgatr_norm_gain_names
    key_map_file = FIX / "key_map.json"
    key_map = json.loads(key_map_file.read_text()) if key_map_file.exists() else {}
    v1 = {key_map.get(name, {}).get(n, n): (tuple(s), rg) for n, s, rg in stored_v1["names"]}
    v2 = {n: (tuple(s), rg) for n, s, rg in live["names"]}
    qkv = {k for k in v1 if QKV_BIAS_RE.search(k)}
    gains = lgatr_norm_gain_names(exp.model)
    assert set(v1) - set(v2) == qkv, (
        f"{name}: removed beyond S5: {sorted((set(v1) - set(v2)) ^ qkv)}")
    added = set(v2) - set(v1)
    assert added == gains, (
        f"{name}: added params are not exactly the structural norm-gain set: "
        f"added-not-gain={sorted(added - gains)} gain-not-added={sorted(gains - added)}")
    drift = {k for k in v2 if k in v1 and v1[k][0] != v2[k][0]}
    assert not drift, f"{name}: shape drift at {sorted(drift)}"
    frozen = _expected_frozen(exp.model)
    flips = {k for k in v2 if k in v1 and v1[k][1] != v2[k][1]}
    assert flips == {k for k in frozen if k in v1 and v1[k][1]}, (
        f"{name}: requires_grad flips beyond the freezing rule: {sorted(flips)}")
    bad_gains = {k for k in gains if v2[k][1] != (k not in frozen)}
    assert not bad_gains, (
        f"{name}: gain requires_grad disagrees with the freezing rule: {sorted(bad_gains)}")
    waived = sum(prod(v1[k][0]) for k in qkv)
    gained = sum(prod(v2[k][0]) for k in gains)
    assert live["total"] == stored_v1["total"] - waived + gained
    print(f"REBASELINE {name}: v1_total={stored_v1['total']} v2_default_total={live['total']} "
          f"(-{waived} qkv biases, +{gained} affine gains in {len(gains)} tensors)")


def _expected_frozen(model):
    """The parameters lgatr 2.0 freezes, recomputed from the module structure -- the Gate B
    requires_grad waiver is this RULE, never a readback of the flags v2 actually set.
    v2's two freezing mechanisms (slim_layers.py / linear.py / layer_norm.py, all tagged
    'zero-size params get grads only sometimes under compile, breaking DDP'):
      1. any zero-size parameter;
      2. _freeze_dead_tail: in an LGATrSlim net whose out_v_channels (out_s_channels) is 0,
         the last block's mlp weight_v (linear_s) parameters and norm2 gain, which sit
         upstream of an empty output stream.
    """
    import lgatr
    frozen = {n for n, p in model.named_parameters() if p.numel() == 0}
    for prefix, mod in model.named_modules():
        if isinstance(mod, lgatr.LGATrSlim) and len(mod.blocks):
            out_v = mod.linear_out.weight_v.shape[0]
            out_s = 0 if mod.linear_out.linear_s is None else mod.linear_out.linear_s.out_features
            last = f"{prefix}.blocks.{len(mod.blocks) - 1}." if prefix else \
                f"blocks.{len(mod.blocks) - 1}."
            for n, _ in model.named_parameters():
                if not n.startswith(last):
                    continue
                tail = n[len(last):]
                if out_v == 0 and ((tail.startswith("mlp.") and tail.endswith("weight_v"))
                                   or tail == "norm2.weight_v"):
                    frozen.add(n)
                if out_s == 0 and ((tail.startswith("mlp.") and "linear_s" in tail)
                                   or tail == "norm2.weight_s"):
                    frozen.add(n)
    return frozen


def _manifest_check_v2(name, stored):
    """Gate B on 2.x: two-sided comparison with waivers derived by rule, never typed by hand.
    Build is PINNED (Phase 3: S1/S2 parity pins) -- under the pins the only legal diffs are
    the S5 qkv-bias removals (derived from the stored v1 names via QKV_BIAS_RE) and the
    requires_grad flips v2's freezing rules produce (derived structurally, _expected_frozen).
    The Posture-B re-baseline at v2 defaults is Task C's own, separate, re-recorded step.
    """
    from collections import Counter
    from math import prod
    key_map_file = FIX / "key_map.json"
    key_map = json.loads(key_map_file.read_text()) if key_map_file.exists() else {}
    v1 = {key_map.get(name, {}).get(n, n): (tuple(s), rg) for n, s, rg in stored["names"]}
    exp = _build("config", PRODUCTION[name] + PARITY_PINS[name], with_data=False)
    live = _manifest(exp.model)
    v2 = {n: (tuple(s), rg) for n, s, rg in live["names"]}
    qkv = {k for k in v1 if QKV_BIAS_RE.search(k)}
    # Two-sided name/shape check: removed == S5 exactly, added == nothing, no shape drift.
    assert set(v1) - set(v2) == qkv and not set(v2) - set(v1), (
        f"{name}: params beyond the S5 rule: removed-not-qkv="
        f"{sorted(set(v1) - set(v2) - qkv)} qkv-not-removed={sorted(qkv - (set(v1) - set(v2)))} "
        f"added={sorted(set(v2) - set(v1))}")
    shape_drift = {k for k in v2 if v1[k][0] != v2[k][0]}
    assert not shape_drift, f"{name}: shape drift at {sorted(shape_drift)}"
    # Two-sided requires_grad check against the structural freezing rule.
    expected_flips = {k for k in _expected_frozen(exp.model) if v1[k][1]}
    flips = {k for k in v2 if v1[k][1] != v2[k][1]}
    assert flips == expected_flips and all(not v2[k][1] for k in flips), (
        f"{name}: requires_grad flips beyond the freezing rule: "
        f"unexpected={sorted(flips - expected_flips)} missing={sorted(expected_flips - flips)}")
    # The recorded Gate B contract: the (shape)|requires_grad multiset, waivers subtracted.
    expect = Counter(stored["entries"])
    expect.subtract(f"{v1[k][0]}|rg={v1[k][1]}" for k in qkv)
    expect.subtract(f"{v1[k][0]}|rg={v1[k][1]}" for k in expected_flips)
    expect.update(f"{v2[k][0]}|rg={v2[k][1]}" for k in expected_flips)
    assert not -expect, f"{name}: waiver subtraction left the v1 multiset negative"
    assert Counter(live["entries"]) == expect, (
        f"{name}: manifest multiset mismatch after rule-derived waivers")
    waived = sum(prod(v1[k][0]) for k in qkv)
    assert live["total"] == stored["total"] - waived
    print(f"GATE-B {name}: v1_total={stored['total']} v2_total={live['total']} "
          f"waived_qkv_biases={len(qkv)} ({waived} params) added=0 "
          f"rg_flips={len(flips)} (rule-matched)")


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


@pytest.mark.parametrize("name", sorted(REDUCED))
def test_config_snapshot_diff(name):
    """Blind spot 2 closure (runbook 4, 'Why these gates discriminate'): eval-mode fixtures
    cannot see train-only config values (a stray `attn_dropout: 0.5` changes no parameter
    and no eval output), so the resolved-config snapshot recorded on 1.4.4 is line-diffed
    against the same composition on the current tree, and the ONLY changes allowed are the
    M-row key renames, value-preserving: M2 `_target_` path shortening (class name must
    survive), M3 `increase_hidden_channels`->`attn_ratio`, M4 `activation`->`nonlinearity`
    and `increase_hidden_channels`->`mlp_ratio`. Anything else -- a changed value, a new
    key, a lost key -- fails here and nowhere else.
    """
    import difflib
    from collections import Counter
    from omegaconf import OmegaConf
    if RECORD or REBASELINE:
        pytest.skip("snapshot diff is a check-mode gate")
    path = FIX / f"{name}.pt"
    if not path.exists():
        pytest.skip("no 1.4.4 fixtures recorded")
    ref = torch.load(path, weights_only=False)
    exp = _build("config_quick", REDUCED[name], with_data=True)
    volatile = ("run_dir:", "run_name:", "run_idx:", "db:", "artifacts:")
    live = "\n".join(l for l in OmegaConf.to_yaml(exp.cfg, resolve=True).splitlines()
                     if not l.strip().startswith(volatile))
    diff = [l for l in difflib.unified_diff(ref["cfg_yaml"].splitlines(), live.splitlines(),
                                            lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    if _lgatr_version().startswith("1.4"):
        assert not diff, f"{name}: config drifted on 1.4.x:\n" + "\n".join(diff)
        return

    def key_of(line):
        return line[1:].strip().split(":", 1)[0]

    def val_of(line):
        parts = line[1:].strip().split(":", 1)
        return parts[1].strip() if len(parts) == 2 else ""

    removed = [l for l in diff if l[0] == "-"]
    added = [l for l in diff if l[0] == "+"]
    bad_removed = [l for l in removed
                   if key_of(l) not in {"increase_hidden_channels", "activation", "_target_"}]
    bad_added = [l for l in added
                 if key_of(l) not in {"attn_ratio", "mlp_ratio", "nonlinearity", "_target_"}]
    assert not bad_removed and not bad_added and len(removed) == len(added), (
        f"{name}: config changes beyond the M-row renames:\n" + "\n".join(diff))
    # renames must be value-preserving: the M rows rename keys, never retune values
    tgt_removed = sorted(val_of(l).rsplit(".", 1)[-1] for l in removed if key_of(l) == "_target_")
    tgt_added = sorted(val_of(l).rsplit(".", 1)[-1] for l in added if key_of(l) == "_target_")
    assert tgt_removed == tgt_added, f"{name}: _target_ class changed: {tgt_removed} -> {tgt_added}"
    vals_removed = Counter(val_of(l) for l in removed if key_of(l) != "_target_")
    vals_added = Counter(val_of(l) for l in added if key_of(l) != "_target_")
    assert vals_removed == vals_added, (
        f"{name}: a rename also changed a value: {dict(vals_removed)} vs {dict(vals_added)}")
    print(f"GATE-CFG {name}: {len(removed)} M-row rename lines, value-preserving")


def test_blade_table_gate_f():
    """Gate F: the hybrids' local Cayley table and lgatr's geometric-product tensor implement
    the same product on the same 16-blade basis (runbook 2.1 'blade layout unchanged' --
    re-proven, not assumed). The index conventions differ by design and the alignment is
    forced, not fitted: hybrid out_j = a_i cayley[i,j,k] b_k (output on the middle axis,
    cliffordalgebra.py); lgatr out_i = gp[i,j,k] x_j y_k (output first, bilinear.py) --
    identical products iff cayley.permute(1, 0, 2) == gp. Bar 2e-6 fp32 (the 1.4.4 audit's
    bar); measured 0.000e+00 on 2.0.0 (both tables hold exact +-1/0 entries, 256 nonzeros).
    """
    try:
        from lgatr.primitives.bilinear import _load_geometric_product_tensor
    except ImportError:
        pytest.skip("Gate F targets lgatr 2.x's geometric-product tensor")
    from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra
    cayley = CliffordAlgebra((1.0, -1.0, -1.0, -1.0)).cayley.to(torch.float32)
    gp = _load_geometric_product_tensor(device=torch.device("cpu"), dtype=torch.float32)
    diff = (cayley.permute(1, 0, 2) - gp).abs().max().item()
    print(f"GATE-F blade-table max|diff|={diff:.3e} "
          f"(nonzeros: hybrid={(cayley != 0).sum().item()} lgatr={(gp != 0).sum().item()})")
    assert diff <= 2e-6, f"blade tables disagree: max|diff|={diff:.3e} > 2e-6"


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
