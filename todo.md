# TODO — outstanding work

A running checklist for finishing the graph-transformer hybrid study and connecting it to
a paper. Grouped by "before training", "open design decisions", and "paper release".

---

## 1. Before training — fill in the training configs

The 8 hybrid recipes are skeletons with required `???` keys:
`config/training/top_{Plain,ParticleNetParT,CGENNLGATr,LorentzNetLGATrSlim}{GraphTrans,GraphGPS}.yaml`.

For each model, fill `iterations`, `batchsize`, `lr` (optionally `weight_decay`, `optimizer`,
`scheduler`):

- [ ] `batchsize` ← `find_lr.py +lr_find.find_batch_size=true` (largest power-of-two that fits the H100).
- [ ] `lr` ← `find_lr.py` (reported loss-min / 10).
- [ ] `iterations` ← `epochs * ceil(num_train_jets / batchsize)` (see §2: pick one epoch budget for all).
- [ ] `weight_decay` ← tune on val ∈ {0, 0.01, 0.05, 0.1} for AdamW (ParT-style 0.01 is a fine start).
- [ ] Decide the shared `scheduler` (see §2) and set it in `tag_default.yaml` (or per-recipe).

## 2. Training-recipe decisions (fairness)

**Scheduler.** Recommend a single shared **cosine-annealing-with-warmup** schedule for the whole
comparison set (hybrids + baselines), tuning only lr/batchsize/weight_decay per model — so the
comparison isolates the architecture, not the recipe. Warmup matters for the transformer-heavy
hybrids. The repo's `CosineAnnealingLR` has no warmup; options:
- [ ] use `OneCycleLR` (`onecycle_pct_start` ≈ 0.05–0.10 gives a short warmup + cosine decay), **or**
- [ ] add a dedicated `LinearWarmup → CosineAnnealingLR` scheduler to `base_experiment._init_scheduler`
      (cleaner "cosine + warmup"; ~5–10% warmup). *(I can add this on request.)*

**Epochs vs iterations.** Everything is configured in *iterations* (`T_max = iterations * scheduler_scale`),
but ParT/ParticleNet calibrate that to ~20 epochs while L-GATr uses a fixed 200k-iter budget. For a
fair comparison fix one **epoch budget** (data exposure) for all models and derive
`iterations = epochs * ceil(num_train_jets / batchsize)` per model; rely on early stopping
(`es_patience`) to stop converged models early. This guarantees the hybrids see the full dataset the
same number of times as the baselines.
- [ ] Pick the epoch budget (e.g. ParT-standard ~20–30 for top-tagging; raise if the hybrids underfit).
- [ ] Re-express the baseline recipes (`top_ParT`, `top_particlenet`, `top_lgatr`) in the same epoch
      budget for the head-to-head table (keep the published-recipe numbers as a separate reference row).

## 3. Open design decisions / discrepancies

- [ ] **CGENN-LGATr GraphGPS local branch has no edge/node features** (`edge_attr_x=0, node_attr_*=0`)
      while its GraphTrans cousin injects raw relative-momentum **edge features** + re-injected raw node
      attributes (standard CGENN/EGNN), and the *other* GraphGPS members (Plain, ParticleNet-ParT) both
      carry edge features in their local MPNN. **Recommend** adding **static** relative-momentum edge
      features (computed once from the raw four-momenta, shared across layers, à la GraphGPS static edge
      attrs) to the CGENN GraphGPS local branch, for a like-for-like local MPNN and a clean
      GraphTrans-vs-GraphGPS ablation. Node-attr re-injection is optional (the hidden mv already carries
      the geometry). *(Analysis in the chat; I can implement.)*
- [ ] Check `LorentzNetLGATrSlimGraphGPS` for the same edge-feature gap vs its GraphTrans cousin.
- [x] CLS readout frame: **jet frame** (covariant, boost into the jet rest frame). Decided.
- [x] LLoCa transport made **strictly additive** (identity frames bit-identical to the plain backbone).

## 4. Paper release — branding / identity (only the maintainer has these)

Critical (still point at the upstream LLoCa project):
- [ ] `README.md` — title ("Lorentz Local Canonicalization"), arXiv badges (2505.20280 / 2508.14898),
      author list + `heidelberg-hepml/*` links, the BibTeX block.
- [ ] `reproduce.md` — clone URL `heidelberg-hepml/lloca-experiments` + `cd lloca-experiments`,
      upstream arXiv references; **replace the manual JetClass-download line with
      `python data/collect_data.py jetclass`** (now automated).
- [ ] `LICENSE` — copyright currently lists the upstream LLoCa authors; add your authors / mark derivative.

Minor (stale strings / metadata):
- [ ] `pyproject.toml` — add an `authors` field (name is already `gtagger-experiments`).
- [ ] add a `CITATION.cff` for the new paper.
- [ ] `experiments/base_experiment.py:262` — `path_code = os.path.join(self.cfg.base_dir, "lloca")`
      hardcodes "lloca" for the saved-source dir → project name.
- [ ] `docs/SLURM.md:79` — `#SBATCH --job-name=lloca`.
- [ ] `config/{toptagging,jctagging,ttbar}.yaml` + `config_quick/*` — debug `exp_name`s
      (`topt_local_debug`, `jc_debug`, `ttbar_debug`).
- [ ] `config/model/tag_CGENNLGATrGraphTrans.yaml` — incomplete `#should be` comment (cosmetic).
- [ ] `tests/helpers/equivariance.py:4` — upstream attribution comment; fine to keep as a credit.
- [ ] **Defork** the GitHub repo when publishing (a fork is hidden from search / awkward to Zenodo-archive);
      keep the upstream attribution in README + LICENSE.

## 5. Done (for reference)

- 2×2×2 hybrid family ({Plain, ParticleNet-ParT, CGENN-LGATr, LorentzNet-LGATr-slim} × {GraphTrans, GraphGPS}).
- Faithful LLoCa tensorial message-passing for the ParticleNet-ParT hybrids (EdgeConv `change_local_frame`
  + `LLoCaAttention`), **additive** (identity frames bit-identical), jet-frame class token, rapidity clamp.
- Equivariance suite (24/24, incl. full Lorentz boost under learned `so(1,3)` frames).
- `find_lr.py` batch-size finder; `aggregate_table.py`; `data/collect_data.py jetclass`; `GUIDE.md`; `docs/SLURM.md`.
