"""Phase -1 of docs/lgatr2-migration.md: verify the runbook's break inventory by RUNNING 2.0.0.

Every M-row and S-row of runbook §2 is checked against observed behavior, not documentation:
a probe subprocess imports lgatr (optionally with a side-loaded 2.0.0 wheel directory prepended
to sys.path, so the installed 1.4.4 environment is untouched), attempts every lgatr construction
this repo performs VERBATIM from its call sites, dumps (name, shape, requires_grad) parameter
lists for tiny nets, and the orchestrator diffs the two runs into a CONFIRMED/WRONG/MISSING
verdict per row.

Usage:
  python tests/experiments/test_lgatr_v2_inventory.py --probe [--prepend DIR] --out FILE
      run the probe under the active lgatr (prepending DIR to sys.path first if given)
  LGATR2_WHEEL_DIR=/path/to/unpacked python -m pytest tests/experiments/test_lgatr_v2_inventory.py
  LGATR2_WHEEL_DIR=... python tests/experiments/test_lgatr_v2_inventory.py
      orchestrate both probes and print the verdict table (pytest skips without the env var,
      so the normal suite never depends on a 2.0.0 wheel being available)

The wheel dir is the UNPACKED wheel (pip download lgatr==2.0.0 --no-deps; unzip), i.e. a
directory containing lgatr/__init__.py. Nothing is installed; the subprocess sys.path trick
leaves the 1.4.4 environment intact for Phase 0's fixture recording.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# probe half: runs once per lgatr version, in its own subprocess
# ---------------------------------------------------------------------------


def _attempt(results, label, fn):
    try:
        out = fn()
        results[label] = {"ok": True, "detail": out if isinstance(out, (str, list, dict, int, float, bool)) else None}
    except Exception as e:  # noqa: BLE001 - the exception TYPE+message is the datum
        results[label] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}


def _params(module):
    return [[n, list(p.shape), bool(p.requires_grad)] for n, p in module.named_parameters()]


def run_probe():
    import torch

    import lgatr

    r = {"version": lgatr.__version__, "file": lgatr.__file__}

    # ---- I: imports, verbatim from repo call sites ----
    _attempt(r, "I1_toplevel", lambda: __import__("lgatr", fromlist=[
        "LGATr", "LGATrSlim", "embed_vector", "extract_scalar", "extract_vector",
        "get_num_spurions", "get_spurions"]) and "ok")
    def i2():  # lorentznetlgatrslimgraphgps.py:50-54 + finetuneexperiment.py:6
        from lgatr.nets.lgatr_slim import MLP, Dropout, Linear, RMSNorm, SelfAttention  # noqa: F401
        return "ok"
    _attempt(r, "I2_v1_slim_path", i2)
    def i3():  # the M1 target path
        from lgatr.layers import SlimDropout, SlimLinear, SlimMLP, SlimRMSNorm, SlimSelfAttention  # noqa: F401
        return "ok"
    _attempt(r, "I3_v2_slim_path", i3)
    def i4():  # hydra _target_: lgatr.nets.lgatr_slim.LGATrSlim (tag/amp/eg_slim.yaml)
        import importlib
        m = importlib.import_module("lgatr.nets.lgatr_slim")
        return "has LGATrSlim" if hasattr(m, "LGATrSlim") else "module without LGATrSlim"
    _attempt(r, "I4_hydra_slim_target", i4)
    def i5():  # wrappers.py flex monkeypatch target
        import lgatr.primitives.attention_backends.flex as flex
        return "attention" if hasattr(flex, "attention") else "NO attention attr"
    _attempt(r, "I5_flex_attention", i5)
    def i6():  # lloca 1.3.6 private-API coupling (H6)
        import lloca.equivectors.lgatr  # noqa: F401
        return "ok"
    _attempt(r, "I6_lloca_import", i6)

    # ---- C: config dataclasses, verbatim kwargs ----
    from lgatr import MLPConfig, SelfAttentionConfig
    _attempt(r, "C1_attn_cfg_v1_names",  # cgennlgatrgraphgps.py:98-99 / tag_lgatr.yaml attention:
             lambda: SelfAttentionConfig(num_heads=2, multi_query=False,
                                         increase_hidden_channels=2, head_scale=True) and "ok")
    _attempt(r, "C2_attn_cfg_v2_names",
             lambda: SelfAttentionConfig(num_heads=2, multi_query=False,
                                         attn_ratio=2, head_scale=True) and "ok")
    _attempt(r, "C3_mlp_cfg_v1_names",  # cgennlgatrgraphgps.py:108-109 / tag_lgatr.yaml mlp:
             lambda: MLPConfig(activation="gelu", increase_hidden_channels=2) and "ok")
    _attempt(r, "C4_mlp_cfg_v2_names",
             lambda: MLPConfig(nonlinearity="gelu", mlp_ratio=2) and "ok")

    # ---- net-level constructions with hydra-style dict subconfigs (H4 loudness probe) ----
    def c5(attn, mlp):
        from lgatr import LGATr
        return LGATr(num_blocks=1, in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=4,
                     in_s_channels=4, out_s_channels=2, hidden_s_channels=8,
                     attention=attn, mlp=mlp)
    _attempt(r, "C5_lgatr_net_v1_dicts", lambda: c5(
        {"num_heads": 2, "multi_query": False, "increase_hidden_channels": 2, "head_scale": True},
        {"activation": "gelu", "increase_hidden_channels": 2}) and "ok")
    _attempt(r, "C6_lgatr_net_v2_dicts", lambda: c5(
        {"num_heads": 2, "multi_query": False, "attn_ratio": 2, "head_scale": True},
        {"nonlinearity": "gelu", "mlp_ratio": 2}) and "ok")

    # ---- L: layer constructions ----
    def l1():  # SelfAttention without primitives (M10)
        from lgatr.layers import SelfAttention
        cfg_kw = dict(num_heads=2, in_mv_channels=2, out_mv_channels=2,
                      in_s_channels=4, out_s_channels=4)
        try:
            cfg = SelfAttentionConfig(attn_ratio=1, **cfg_kw)
        except TypeError:
            cfg = SelfAttentionConfig(increase_hidden_channels=1, **cfg_kw)
        return SelfAttention(cfg) and "ok"
    _attempt(r, "L1_selfattention_no_primitives", l1)
    def l2():  # GeoMLP without primitives (M10)
        from lgatr.layers import GeoMLP
        try:
            cfg = MLPConfig(mv_channels=2, s_channels=4, nonlinearity="gelu", mlp_ratio=2)
        except TypeError:
            cfg = MLPConfig(mv_channels=2, s_channels=4, activation="gelu", increase_hidden_channels=2)
        return GeoMLP(cfg) and "ok"
    _attempt(r, "L2_geomlp_no_primitives", l2)
    def l3():  # EquiLinear without primitives (M10: third positional on v2)
        from lgatr.layers import EquiLinear
        return EquiLinear(4, 4, in_s_channels=4, out_s_channels=4) and "ok"
    _attempt(r, "L3_equilinear_no_primitives", l3)
    def l4():  # the v2 forms, with PrimitivesConfig
        from lgatr import PrimitivesConfig
        from lgatr.layers import EquiLinear, GeoMLP, SelfAttention
        prim = PrimitivesConfig()
        cfg = SelfAttentionConfig(num_heads=2, in_mv_channels=2, out_mv_channels=2,
                                  in_s_channels=4, out_s_channels=4, attn_ratio=1)
        SelfAttention(cfg, prim)
        GeoMLP(MLPConfig(mv_channels=2, s_channels=4, nonlinearity="gelu", mlp_ratio=2), prim)
        EquiLinear(4, 4, prim, in_s_channels=4, out_s_channels=4)
        return "ok"
    _attempt(r, "L4_layers_with_primitives", l4)
    def l5():  # bare RMSNorm() -- lorentznetlgatrslimgraphgps.py:91 (M5)
        try:
            from lgatr.nets.lgatr_slim import RMSNorm as N
        except ImportError:
            from lgatr.layers import SlimRMSNorm as N
        return N() and "ok"
    _attempt(r, "L5_bare_slim_rmsnorm", l5)
    def _slim_names():
        try:
            from lgatr.nets.lgatr_slim import MLP, Linear, SelfAttention  # v1
            return SelfAttention, MLP, Linear
        except ImportError:
            from lgatr.layers import SlimLinear, SlimMLP, SlimSelfAttention  # v2
            return SlimSelfAttention, SlimMLP, SlimLinear
    def l6():  # verbatim GPS attention construction (lorentznetlgatrslimgraphgps.py:81-85)
        SA, _, _ = _slim_names()
        return SA(v_channels=6, s_channels=16, num_heads=2, attn_ratio=1, dropout_prob=None) and "ok"
    _attempt(r, "L6_slim_attention_verbatim", l6)
    def l7():  # verbatim GPS mlp construction (:86-90)
        _, MLP_, _ = _slim_names()
        return MLP_(v_channels=6, s_channels=16, nonlinearity="gelu",
                    mlp_ratio=2, num_layers=2, dropout_prob=None) and "ok"
    _attempt(r, "L7_slim_mlp_verbatim", l7)
    def l8():  # verbatim linear_in construction (:175)
        _, _, Lin = _slim_names()
        return Lin(in_v_channels=2, out_v_channels=6, in_s_channels=5, out_s_channels=16) and "ok"
    _attempt(r, "L8_slim_linear_verbatim", l8)
    def l9():  # bare EquiLayerNorm() -- cgennlgatrgraphgps.py per-layer norms (S2 layer default)
        from lgatr.layers import EquiLayerNorm
        n = EquiLayerNorm()
        return f"params={sum(p.numel() for p in n.parameters())}"
    _attempt(r, "L9_bare_equilayernorm", l9)
    def l11():  # M6: literal None s-channels on a slim linear
        _, _, Lin = _slim_names()
        return Lin(in_v_channels=2, out_v_channels=6, in_s_channels=None, out_s_channels=None) and "ok"
    _attempt(r, "L11_slim_linear_none_schannels", l11)

    # ---- B: behavior probes ----
    from lgatr import embed_vector, get_spurions
    _attempt(r, "B1_embed_slots",
             lambda: embed_vector(torch.tensor([1.0, 2.0, 3.0, 4.0])).nonzero().flatten().tolist())
    _attempt(r, "B2_spurions",  # verbatim spurion kwargs used by the hybrids
             lambda: get_spurions(beam_spurion="xyplane", add_time_spurion=True,
                                  beam_mirror=True).flatten().tolist())
    def b3():
        from lgatr import PrimitivesConfig
        c = PrimitivesConfig()
        return f"subgroup={c.subgroup} sparse_gp={c.sparse_gp} sparse_linear={c.sparse_linear}"
    _attempt(r, "B3_primitives_defaults", b3)
    def b4():
        import inspect
        from lgatr import LGATrSlim
        sig = inspect.signature(LGATrSlim.__init__)
        keys = ["nonlinearity_v", "norm_elementwise_affine", "naive_amp",
                "compile", "compile_kwargs", "compile_mode", "compile_dynamic"]
        return {k: (repr(sig.parameters[k].default) if k in sig.parameters else "ABSENT") for k in keys}
    _attempt(r, "B4_slim_signature", b4)
    def _tiny_slim(**kw):
        from lgatr import LGATrSlim
        return LGATrSlim(in_v_channels=1, out_v_channels=0, hidden_v_channels=6,
                         in_s_channels=5, out_s_channels=2, hidden_s_channels=16,
                         num_blocks=1, num_heads=2, **kw)
    def b5():  # M8: net-level interface is channel-FIRST on both versions
        m = _tiny_slim().double().eval()
        v = torch.randn(2, 7, 1, 4, dtype=torch.float64)
        s = torch.randn(2, 7, 5, dtype=torch.float64)
        out_v, out_s = m(v, s)
        return f"net_forward ok, out_s {list(out_s.shape)}"
    _attempt(r, "B5_slim_net_channel_first", b5)
    def b5b():  # M8: BLOCK-level layout -- which orientation SlimLinear/Linear accepts
        _, _, Lin = _slim_names()
        lin = Lin(in_v_channels=3, out_v_channels=6, in_s_channels=5, out_s_channels=7).double()
        s = torch.randn(2, 5, dtype=torch.float64)
        res = {}
        for tag, shape in [("channel_first_(C,4)", (2, 3, 4)), ("channel_last_(4,C)", (2, 4, 3))]:
            try:
                ov, _ = lin(torch.randn(*shape, dtype=torch.float64), s)
                res[tag] = f"ok out {list(ov.shape)}"
            except Exception as e:  # noqa: BLE001
                res[tag] = f"{type(e).__name__}"
        return res
    _attempt(r, "B5b_slim_linear_layout", b5b)
    _attempt(r, "B6_tiny_slim_params_default", lambda: _params(_tiny_slim()))
    def b6b():
        try:
            return _params(_tiny_slim(norm_elementwise_affine=False, nonlinearity_v=None))
        except TypeError as e:
            return f"kwargs absent: {e}"
    _attempt(r, "B6b_tiny_slim_params_pinned", b6b)
    def b7():
        from lgatr import LGATr
        try:
            m = c5({"num_heads": 2, "multi_query": False, "attn_ratio": 1, "head_scale": True},
                   {"nonlinearity": "gelu", "mlp_ratio": 2})
        except TypeError:
            m = c5({"num_heads": 2, "multi_query": False, "increase_hidden_channels": 1,
                    "head_scale": True}, {"activation": "gelu", "increase_hidden_channels": 2})
        return _params(m)
    _attempt(r, "B7_tiny_lgatr_params", b7)
    def b9():  # S6: the GLU vector-gate 0.5 = 1/sqrt(4) scale, from source
        import inspect
        try:
            from lgatr.layers.slim_layers import SlimGLU as GLU
        except ImportError:
            from lgatr.nets.lgatr_slim import GatedLinearUnit as GLU
        src = inspect.getsource(GLU)
        return f"has_0.5_scale={'0.5' in src and 'sqrt(4)' in src}"
    _attempt(r, "B9_glu_scale", b9)
    def b12():  # sparse_gp custom backward sanity (v2 only): gradcheck in fp64
        from lgatr import PrimitivesConfig
        from lgatr.primitives.bilinear import geometric_product
        cfg = PrimitivesConfig(sparse_gp=True)
        a = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
        b = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
        ok = torch.autograd.gradcheck(lambda x, y: geometric_product(x, y, config=cfg), (a, b))
        return f"gradcheck={ok}"
    _attempt(r, "B12_sparse_gp_gradcheck", b12)

    # ---- R: repo model composition per config (which configs build at all) ----
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import logging.handlers  # noqa: F401  (experiments.logger assumes it is already imported)
    import experiments.logger
    experiments.logger.LOGGER.disabled = True
    import hydra
    from experiments.tagging.experiment import TopTaggingExperiment
    MODELS = ["tag_lgatr", "tag_slim", "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS",
              "tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS"]
    for name in MODELS:
        def build(name=name):
            with hydra.initialize(config_path="../../config_quick", version_base=None):
                cfg = hydra.compose(config_name="toptagging",
                                    overrides=["save=false", f"model={name}"])
            exp = TopTaggingExperiment(cfg)
            exp._init()
            exp.init_physics()
            exp.init_model()
            return f"built, params={sum(p.numel() for p in exp.model.parameters())}"
        _attempt(r, f"R_{name}", build)

    return r


# ---------------------------------------------------------------------------
# orchestrator half: run the probe under 1.4.4 and under the side-loaded wheel
# ---------------------------------------------------------------------------


def _run_subprocess(prepend, out_path):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--probe", "--out", str(out_path)]
    if prepend:
        cmd += ["--prepend", str(prepend)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return json.loads(Path(out_path).read_text())


def _verdicts(v1, v2):
    """Map probe observations onto the runbook's M/S rows."""
    def ok(r, k):
        return r.get(k, {}).get("ok")
    def det(r, k):
        return r.get(k, {}).get("detail")
    rows = []
    def row(rid, claim, confirmed, evidence):
        rows.append((rid, "CONFIRMED" if confirmed else "WRONG/CHECK", claim, evidence))

    row("M1", "v1 slim import path dies; Slim* names live in lgatr.layers on v2",
        ok(v1, "I2_v1_slim_path") and not ok(v2, "I2_v1_slim_path")
        and not ok(v1, "I3_v2_slim_path") and ok(v2, "I3_v2_slim_path"),
        f"v1 I2 {ok(v1,'I2_v1_slim_path')} I3 {ok(v1,'I3_v2_slim_path')} | "
        f"v2 I2 {ok(v2,'I2_v1_slim_path')} I3 {ok(v2,'I3_v2_slim_path')}")
    row("M2", "hydra target lgatr.nets.lgatr_slim.LGATrSlim gone on v2; top-level LGATrSlim fine",
        ok(v1, "I4_hydra_slim_target") and not ok(v2, "I4_hydra_slim_target")
        and ok(v2, "I1_toplevel"),
        f"v2 I4: {det(v2,'I4_hydra_slim_target')}")
    row("M3", "SelfAttentionConfig increase_hidden_channels -> attn_ratio, old name RAISES",
        ok(v1, "C1_attn_cfg_v1_names") and not ok(v2, "C1_attn_cfg_v1_names")
        and ok(v2, "C2_attn_cfg_v2_names"),
        f"v2 C1: {det(v2,'C1_attn_cfg_v1_names')}")
    row("M4+H4", "MLPConfig renames; hydra-style dict cast raises LOUDLY on stale keys",
        ok(v1, "C3_mlp_cfg_v1_names") and not ok(v2, "C3_mlp_cfg_v1_names")
        and ok(v2, "C4_mlp_cfg_v2_names")
        and ok(v1, "C5_lgatr_net_v1_dicts") and not ok(v2, "C5_lgatr_net_v1_dicts")
        and ok(v2, "C6_lgatr_net_v2_dicts"),
        f"v2 C5: {det(v2,'C5_lgatr_net_v1_dicts')}")
    row("M5", "bare slim RMSNorm() raises on v2",
        ok(v1, "L5_bare_slim_rmsnorm") and not ok(v2, "L5_bare_slim_rmsnorm"),
        f"v2 L5: {det(v2,'L5_bare_slim_rmsnorm')}")
    # Observed (2026-08): the runbook's original M6 direction was BACKWARDS -- v1.4.4 RAISES on a
    # literal None s-channel (nn.Linear(None, ...) TypeError at construction) while v2 tolerates
    # it. Since init_physics always fills real ints, no live path ever passed None on either
    # version -> non-break; the row is kept to pin the observed asymmetry.
    row("M6*", "literal None s-channels: v1 RAISES, v2 tolerates (runbook direction corrected)",
        not ok(v1, "L11_slim_linear_none_schannels") and ok(v2, "L11_slim_linear_none_schannels"),
        f"v1: {det(v1,'L11_slim_linear_none_schannels')} | v2: ok")
    d1, d2 = det(v1, "B5b_slim_linear_layout"), det(v2, "B5b_slim_linear_layout")
    d1, d2 = d1 if isinstance(d1, dict) else {}, d2 if isinstance(d2, dict) else {}
    row("M8", "net-level slim forward stays channel-first on BOTH; block Linear flips to channel-last on v2",
        ok(v1, "B5_slim_net_channel_first") and ok(v2, "B5_slim_net_channel_first")
        and str(d1.get("channel_first_(C,4)", "")).startswith("ok")
        and not str(d1.get("channel_last_(4,C)", "")).startswith("ok")
        and str(d2.get("channel_last_(4,C)", "")).startswith("ok")
        and not str(d2.get("channel_first_(C,4)", "")).startswith("ok"),
        f"v1 block: {d1} | v2 block: {d2}")
    v1sig = det(v1, "B4_slim_signature") if isinstance(det(v1, "B4_slim_signature"), dict) else {}
    row("M9", "slim compile kwargs: v1 compile_mode/compile_dynamic -> v2 compile_kwargs",
        v1sig.get("compile_mode", "ABSENT") != "ABSENT"
        and isinstance(det(v2, "B4_slim_signature"), dict)
        and det(v2, "B4_slim_signature").get("compile_mode") == "ABSENT"
        and det(v2, "B4_slim_signature").get("compile_kwargs") != "ABSENT",
        f"v1 {v1sig} | v2 {det(v2,'B4_slim_signature')}")
    row("M10", "SelfAttention/GeoMLP/EquiLinear need primitives on v2 (TypeError without)",
        ok(v1, "L1_selfattention_no_primitives") and ok(v1, "L2_geomlp_no_primitives")
        and ok(v1, "L3_equilinear_no_primitives")
        and not ok(v2, "L1_selfattention_no_primitives") and not ok(v2, "L2_geomlp_no_primitives")
        and not ok(v2, "L3_equilinear_no_primitives") and ok(v2, "L4_layers_with_primitives"),
        f"v2 L1/L2/L3: {det(v2,'L1_selfattention_no_primitives')} | "
        f"{det(v2,'L2_geomlp_no_primitives')} | {det(v2,'L3_equilinear_no_primitives')}")
    v2sig = det(v2, "B4_slim_signature") if isinstance(det(v2, "B4_slim_signature"), dict) else {}
    row("S1", "nonlinearity_v defaults to 'sigmoid' on v2 (absent on v1)",
        v1sig.get("nonlinearity_v") == "ABSENT" and v2sig.get("nonlinearity_v") == "'sigmoid'",
        f"v1 nonlinearity_v={v1sig.get('nonlinearity_v')} v2 nonlinearity_v={v2sig.get('nonlinearity_v')}")
    def gains(plist):
        # affine NORM gains only: slim Linear's main weight is ALSO named weight_v on both
        # versions, so the suffix alone cannot identify a gain -- require a norm module path
        return sorted(n for n, s, _ in plist
                      if n.split(".")[-1] in ("weight_v", "weight_s", "weight_mv")
                      and ".norm" in "." + n)
    v1p, v2p = det(v1, "B6_tiny_slim_params_default"), det(v2, "B6_tiny_slim_params_default")
    row("S2", "norm affine gains appear by default on v2 nets, absent on v1; pinned-off restores",
        isinstance(v1p, list) and isinstance(v2p, list)
        and not gains(v1p) and bool(gains(v2p))
        and isinstance(det(v2, "B6b_tiny_slim_params_pinned"), list)
        and not gains(det(v2, "B6b_tiny_slim_params_pinned")),
        f"v1 norm gains: {gains(v1p) if isinstance(v1p, list) else v1p} | "
        f"v2 norm gains: {gains(v2p) if isinstance(v2p, list) else v2p}")
    fp1, fp2 = det(v1, "B7_tiny_lgatr_params"), det(v2, "B7_tiny_lgatr_params")
    def mvgains(plist):
        return sorted((n, tuple(s)) for n, s, _ in plist
                      if n.endswith("weight_mv") and ".norm" in "." + n)
    row("S2-full", "full-LGATr norm gains are per-grade (mv_channels, 5) on v2, absent on v1",
        isinstance(fp1, list) and isinstance(fp2, list) and not mvgains(fp1)
        and bool(mvgains(fp2)) and all(s[-1] == 5 for _, s in mvgains(fp2)),
        f"v2 mv gains: {mvgains(fp2)[:3] if isinstance(fp2, list) else fp2}")
    def biases(plist):
        return sorted(n for n, s, _ in plist if n.endswith("bias") and (".linear_in." in n or "qkv" in n))
    row("S5", "qkv-projection biases exist on v1, absent on v2 (slim + full)",
        isinstance(v1p, list) and bool(biases(v1p)) and isinstance(v2p, list) and not biases(v2p),
        f"v1 qkv biases: {biases(v1p) if isinstance(v1p, list) else v1p} | "
        f"v2: {biases(v2p) if isinstance(v2p, list) else v2p}")
    row("S3+", "PrimitivesConfig defaults sparse_gp=True subgroup=True; gradcheck passes",
        "sparse_gp=True" in str(det(v2, "B3_primitives_defaults"))
        and "subgroup=True" in str(det(v2, "B3_primitives_defaults"))
        and "gradcheck=True" in str(det(v2, "B12_sparse_gp_gradcheck")),
        f"{det(v2,'B3_primitives_defaults')}; {det(v2,'B12_sparse_gp_gradcheck')}")
    row("S6", "GLU 0.5=1/sqrt(4) scale in v2 source, absent in v1",
        "has_0.5_scale=False" in str(det(v1, "B9_glu_scale"))
        and "has_0.5_scale=True" in str(det(v2, "B9_glu_scale")),
        f"v1 {det(v1,'B9_glu_scale')} v2 {det(v2,'B9_glu_scale')}")
    row("NB-blade", "embed_vector slots 1:5 on both (blade layout unchanged)",
        det(v1, "B1_embed_slots") == [1, 2, 3, 4] == det(v2, "B1_embed_slots"),
        f"v1 {det(v1,'B1_embed_slots')} v2 {det(v2,'B1_embed_slots')}")
    row("NB-spurions", "spurion values byte-identical",
        det(v1, "B2_spurions") == det(v2, "B2_spurions"),
        "equal" if det(v1, "B2_spurions") == det(v2, "B2_spurions") else "DIFFER")
    row("NB-layer-norm", "bare EquiLayerNorm() stays valid and parameter-free on v2",
        ok(v2, "L9_bare_equilayernorm") and "params=0" in str(det(v2, "L9_bare_equilayernorm")),
        f"v2 L9: {det(v2,'L9_bare_equilayernorm')}")
    row("NB-flex+lloca", "flex.attention attr and lloca import survive v2",
        ok(v2, "I5_flex_attention") and ok(v2, "I6_lloca_import"),
        f"flex {det(v2,'I5_flex_attention')} lloca {ok(v2,'I6_lloca_import')}")
    row("NB-slim-blocks", "verbatim slim block constructions arg-compatible on v2",
        ok(v2, "L6_slim_attention_verbatim") and ok(v2, "L7_slim_mlp_verbatim")
        and ok(v2, "L8_slim_linear_verbatim"),
        f"L6/7/8 v2: {ok(v2,'L6_slim_attention_verbatim')}/{ok(v2,'L7_slim_mlp_verbatim')}/{ok(v2,'L8_slim_linear_verbatim')}")
    for name in ["tag_lgatr", "tag_slim", "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS",
                 "tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS"]:
        k = f"R_{name}"
        rows.append((k, "v1 " + ("OK" if ok(v1, k) else "FAIL") + " / v2 "
                     + ("OK(!)" if ok(v2, k) else "FAIL(expected pre-port)"),
                     "repo composition", str(det(v2, k))[:110]))
    return rows


def orchestrate(wheel_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v1 = _run_subprocess(None, out_dir / "probe_v1.json")
    v2 = _run_subprocess(wheel_dir, out_dir / "probe_v2.json")
    assert v1["version"].startswith("1.4"), f"baseline probe ran {v1['version']}, expected 1.4.x"
    assert v2["version"] == "2.0.0", f"side-load probe ran {v2['version']}, expected 2.0.0"
    rows = _verdicts(v1, v2)
    width = max(len(r[0]) for r in rows)
    lines = [f"lgatr inventory: v1={v1['version']}  v2={v2['version']} (side-loaded)"]
    for rid, verdict, claim, evidence in rows:
        lines.append(f"{rid:<{width}}  {verdict:<26}  {claim}")
        lines.append(f"{'':<{width}}      evidence: {evidence}")
    report = "\n".join(lines)
    (out_dir / "verdicts.txt").write_text(report)
    return report, rows


def test_v2_inventory():
    """Pytest entry: needs an unpacked 2.0.0 wheel dir via LGATR2_WHEEL_DIR (else skip)."""
    import pytest
    wheel_dir = os.environ.get("LGATR2_WHEEL_DIR")
    if not wheel_dir:
        pytest.skip("set LGATR2_WHEEL_DIR to an unpacked lgatr-2.0.0 wheel directory")
    report, rows = orchestrate(wheel_dir, Path(wheel_dir).parent / "inventory_out")
    print(report)
    bad = [r for r in rows if r[1] == "WRONG/CHECK"]
    assert not bad, "inventory rows not confirmed:\n" + "\n".join(str(b) for b in bad)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--prepend", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.probe:
        if args.prepend:
            sys.path.insert(0, args.prepend)
        res = run_probe()
        text = json.dumps(res, indent=1)
        if args.out:
            Path(args.out).write_text(text)
        else:
            print(text)
    else:
        wheel_dir = os.environ.get("LGATR2_WHEEL_DIR")
        assert wheel_dir, "set LGATR2_WHEEL_DIR"
        report, _ = orchestrate(wheel_dir, Path(wheel_dir).parent / "inventory_out")
        print(report)
