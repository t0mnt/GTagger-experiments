# lloca 1.3.6 → 2.0.0 migration

Companion to `docs/lgatr2-migration.md`; same posture (one variable at a time, pins first).
lgatr stays at 2.0.x. kingdon is untouched here (see the kingdon note at the end).

## What 2.0 changes for this fork

Verified against the `heidelberg-hepml/lloca` 2.0.0 tag (commit `27e3db1`), by diffing the
constructor and forward signatures of every symbol this repo imports and by running the
CPU test tree against a 2.0 install before and after the port.

Breaks (each one reproduced by a test before the fix):

- `LLoCaAttention.__init__` gained `preserve_variance=True` (default). With it on,
  `prepare_frames` requires the reference momentum `p_ref` and raises otherwise. All five
  transport sites here call `prepare_frames(frames)` without it, and the library
  `lloca.backbone.transformer.Transformer` (amp/tag/eg `*_transformer.yaml`) has the same
  default and the same `ValueError` from its `forward`.
- `LearnedFrames.__init__` no longer takes `compile` (the FramesNet `compile` option was
  removed) and `LearnedPDFrames` no longer takes the deprecated `deterministic_boost`.
  Hydra passes both from `config/model/framesnet/*.yaml`, so every learned-frames config
  failed at instantiation (`TypeError`).
- `lloca.backbone.particletransformer.ParticleTransformer` asserts `not trim` when used
  with LLoCa frames (the trimmer permutes tokens but not frames -- the bug this fork's
  vendored ParT already fixed locally). The identity-path parity pin constructed it with
  `trim=True`.
- `torch>=2.4` is now required (`nn.RMSNorm`), and `einops` is no longer imported by
  lloca, so the transitive pin added during the lgatr 2.0 migration goes away.
- **ParT per-head scaling changed numerics.** lloca 2.0 pulls weaver-core main
  (`d53f590`), and weaver main replaced the original `torch.einsum("bthd,h->btdh", x,
  c_attn)` with `x * c_attn.view(1, 1, H, 1)`. The original permutes (head, dim) before the
  reshape back to `embed_dim`; the new one does not. Both are followed by the same
  `out_proj`, so this is a different function, not a layout no-op -- forward outputs
  differ by O(1) at init (`c_attn` starts at ones, where the new form is the identity and
  the old form is a fixed permutation). Localized by hooking the first block: embed,
  pair-embed, attention output and both masks are bit-identical between 1.3.6 and 2.0;
  the first divergence is the `post_attn_norm` input, and it vanishes with
  `scale_heads=False` in both the particle and the class blocks. The vendored
  `experiments/baselines/particletransformer.py` keeps the einsum, because that is what
  every stored ParT checkpoint was trained with.

Confirmed non-breaks: every other import resolves unchanged (`TensorReps`,
`TensorRepsTransform`, `Frames` family, `change_local_frame`, `restframe_boost`,
`rand_*`, `get_*_from_*`, `_canonical_mask`, `pairwise_lv_fts_pp`, `MLP`, the flex
backend module). `get_batch_from_ptr`/`get_ptr_from_batch` gained optional
`num_items`/`num_graphs` arguments (sync-free path); the old positional call still works.
`lloca_transport_attention` only uses `LLoCaAttention.forward`, which is unchanged.

## Decisions

Two commits. The first is the bit-exact port (below). The second turns on the 2.0
improvements that are worth a retrain:

- **`preserve_variance=True` everywhere** (five nets, seven transformer configs; a real
  constructor argument on every net, no `**kwargs` swallowing). The base wrapper computes
  `p_ref` once per event as the global-frame, energy-first jet momentum over real particles
  and every wrapper passes it. Class tokens (GraphTrans hybrids) already used the covariant
  jet rest frame (`cls_frames=self._jet_frames`); the transformer's global readout tokens now
  do too (`compute_jet_frames = not mean_aggregation`), because an identity frame there
  gives gamma = E_jet/m_jet in the lab, which breaks invariance under boosts. In the jet
  rest frame gamma is 1. Pinned by `tests/experiments/test_preserve_variance.py`: p_ref
  reaches `LLoCaAttention`, gamma >= 1 with max > 1.05 on top jets, readout gamma ~ 1, the
  flag changes the score, Lorentz invariance holds (learnedpd float64 tolerance).
- **Amplitudes** pass `p_ref = fourmomenta_global.sum(-2)`, the total momentum of the
  process (all energies positive, incoming partons included; timelike with E > 0 for every
  event in `zgggg`/`ttbar`). **Event generation stays at `preserve_variance=False`**: the
  CFM velocity net sees the interpolated `x(t)`, whose event total is noise-dominated at
  small `t` (not timelike, energy sign not guaranteed), so there is no sound reference
  momentum. Turning it on there would need a schedule-aware reference (e.g. the data
  endpoint's total momentum), which is its own experiment.
- **Numerics:** with the rescaling on, the learnedso13 Lorentz-invariance floor of the
  hybrids drops (PlainGraphGPS, 4 seeds: 4e-7 to 5e-6 versus 1e-5 to 2e-4 with it off),
  because the transported q/k/v no longer carry gamma^grade amplification. The
  `test_lloca_frame_invariance` assert tolerance (1e-4) sits below the file's stated 1e-3
  static-kNN neighbour-flip floor, and one RNG-order-dependent PlainGraphGPS case crossed
  it: replayed with the same init and batch, 3 of 2960 node-slots re-ranked under the boosts
  with the flag on and off alike, gamma_i was stable to 9e-5, and the score moved 3.9e-4 (on)
  versus 1.7e-4 (off). The flip is the cause; the flag only scales its footprint. The test
  now runs PlainGraphGPS fully connected, as it already did for the dynamic-kNN hybrid.
- **`p_ref` is cast to the network dtype** (the frames already were). Production runs
  float64 momenta (`data.momentum_float64: true`) through float32 nets; a float64 `p_ref`
  makes lloca fold a float64 gamma into the frames and silently run the whole q/k/v
  transport in float64 (verified on `LLoCaAttention` directly: `frames_out` came back
  float64 while the output stayed float32). Pinned by `test_p_ref_keeps_the_network_dtype`.
- **weaver-main per-head scale in the vendored ParT** (`legacy_head_scale=False` default).
  The original einsum permuted (head, dim) before `out_proj`, so `scale_heads` never was a
  per-head gain. Checkpoints trained with the einsum load bit-exactly with
  `block_params=dict(legacy_head_scale=True)` (and the same in `cls_block_params`). The
  identity-path parity pin against the library runs with the head scale on again.
- Not changed: `trim=True` stays on the official ParT row only (efficiency plus a weak
  random-truncation augmentation on the longest events); hybrids never trimmed.

The bit-exact port:

- **`preserve_variance=False` everywhere, explicitly** (five code sites, seven transformer configs). This is the 1.3.6 numerics, so every
  recorded baseline and checkpoint stays valid. lloca's own docs take the same route for the
  ParT/transformer ports ("pass `preserve_variance=False` to keep the diff minimal").
  Turning it on is a real architecture change (per-token 1/gamma rescaling of q/k/v and the
  output) and needs `p_ref` = the jet four-momentum threaded from the wrapper into
  `prepare_frames` (dense: `(B, 4)`; packed: `(num_jets, 4)` plus `ptr`). Do that as its
  own config flag with its own gates, not inside this port.
- **Config keys removed, not renamed.** `compile: false` was a no-op already (framesnet
  compile was never turned on here). `deterministic_boost: null` was the deprecated
  default.
- **Parity pin runs `trim=False` and `scale_heads=False` (particle and class blocks).** The
  pin is for the identity-frames path; the trimmer was never what it pinned, and the
  per-head scale is now a documented divergence from the library (above). With both off the
  pin is bit-exact again, so it still guards embedding, pair embedding, masks, attention,
  FFN, class attention and the head.
- **Not adopting weaver's new `c_attn` form.** Doing so would silently re-weight every
  stored ParT checkpoint (the permutation is baked into their `out_proj`). If a future
  retrain wants the new form, it is one line in the vendored Block plus a fresh baseline.

## Follow-ups (not in this port)

- Port lloca 2.0's ParT fix "compile the forward, not the whole class" into the vendored
  `experiments/baselines/particletransformer.py` (`torch.compile(self.__class__, ...)`).
- kingdon: 3.0.0 is out. `utils/flash_gen.py` regenerates `flash_ref_p1m3.py` byte-for-byte
  identical under 3.0.0 (only the version stamps differ) and the Step 1 gates in
  `tests/internal/test_kingdon_conventions.py` pass except the deliberate version pin.
  Bumped in its own commit after re-running the gates. kingdon is deliberately not in
  `requirements.txt`: it is a codegen/gate tool, and the gates `importorskip` it.
