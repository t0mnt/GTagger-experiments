"""The GPU half of the compile workflow (docs/cgenn-compile.md, "Upstream's compile
workflow"): profile REAL training steps and print the two lists that section defines.

    python utils/profile_sync.py -cn toptagging model=tag_PlainGraphTrans save=false \
        training.batchsize=<sized>
    python utils/profile_sync.py -cn toptagging model=tag_cgenn save=false \
        training.batchsize=<sized>
    python utils/profile_sync.py -cn toptagging model=tag_CGENNLGATrGraphGPS save=false \
        training.batchsize=<sized>

Tunables (absent from the base config, so the `+` prefix is REQUIRED):
    +prof.warm=8      full training steps before the profiler (compile/autotune land here)
    +prof.active=5    steady-state steps recorded (window = 1 wait + 2 warmup + active)
    +prof.memory=true also record allocator events (adds overhead; off by default --
                      the VRAM question belongs to utils/vram_compile_matrix.py)

Run it INSIDE a GPU allocation (`interact -g 1`, then apptainer + venv as in
docs/OSCAR.md); on a CPU-only node it still runs so the harness can be smoke-tested, but
the sync hunt it exists for needs the card and it says so loudly.

What it does, and the three mistakes it exists to prevent (each shipped in an earlier
revision of the recipe):

  * Warm-up happens OUTSIDE the profiler (8 full `_step`s: compilation, autotune,
    allocator growth, the ParT trimmer's 5-step warmup), then a
    `schedule(wait=1, warmup=2, active=5)` window records 5 steady-state steps.
  * It profiles `exp._step`, not `_batch_loss` + `backward` -- the per-step DRIVER reads
    (`loss.item()`, the non-finite guard's `isfinite`, tracker `.cpu().item()`s) are
    sync-hunt targets and only exist at `_step` scope.
  * STEP 1 (sync hunt) reads CPU-side rows: the explicit stall list (cudaStreamSynchronize
    / Memcpy / aten::item / _local_scalar_dense) plus the `self_cpu_time_total` table with
    stacks, so each stall names its python line. STEP 2 (hot kernels) is the
    `self_cuda_time_total` table, printed second. Sorting by CUDA time first was the old
    recipe's mistake -- it ranks kernels and can never surface a stall.

The chrome trace lands next to the tables (profiles/sync_<model>.json.gz): open in
chrome://tracing or perfetto and read the GPU row's GAPS -- a drained queue between kernel
spans is the launch-bound signature the census predicts (~500-550 launches/step on
production CGENN-GraphGPS).

Batchsize: pass the SIZED one (`utils/vram_compile_matrix.py` / `find_lr`) -- an unswept
recipe merges the 512 fallback, and a profile at a batch nobody trains at answers a
question nobody asked. The compile posture is left to the yaml on purpose: the shipped
posture is the thing to profile.
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# dynamo re-warns this once per traced lru_cache call site; on the CGENN hybrids that is
# ~100 identical lines drowning the tables this tool exists to print
warnings.filterwarnings("ignore", message=".*functools.lru_cache.*")

import hydra
import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, schedule

from experiments.logger import LOGGER
from utils.find_lr import CONSTRUCTORS, _cycle, build_experiment

WARM_STEPS = 8            # compile + autotune + trimmer warm-up, all outside the window
WAIT, WARMUP = 1, 2
ACTIVE = 5

# Explicit stall signatures for the step-1 shortlist. Substring match on the event key;
# `_local_scalar_dense` is what `.item()` and a tensor-as-bool lower to.
STALL_KEYS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy", "Memcpy",
              "aten::item", "aten::_local_scalar_dense", "cudaStreamWaitEvent")


@hydra.main(config_path="../config", config_name="toptagging", version_base=None)
def main(cfg):
    # dynamo re-warns for EVERY traced lru_cache call -- hundreds per compile, drowning
    # the tables this tool exists to print (same "once" treatment as utils/find_lr.py)
    import warnings
    warnings.filterwarnings("once", message=".*lru_cache.*")
    cfg.train = False
    cfg.evaluate = False
    cfg.plot = False
    cfg.save = False

    cuda = torch.cuda.is_available()
    if not cuda:
        LOGGER.warning(
            "No CUDA device: running CPU-only so the harness can be checked, but the "
            "sync hunt this tool exists for is only meaningful on the GPU allocation."
        )

    # the production task configs derive `iterations` from `epochs` inside train(), which
    # never runs here; the scheduler only needs a horizon, and the LR value cannot move a
    # sync or a kernel, so any finite one serves
    if OmegaConf.select(cfg, "training.iterations", default=None) is None:
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.training.iterations = 10_000

    exp = build_experiment(cfg)
    exp._init_scheduler()
    # the scaffolding train() would provide -- _step appends into these
    exp.train_lr, exp.train_loss, exp.val_loss = [], [], []
    exp.grad_norm_train, exp.grad_norm_frames, exp.grad_norm_net = [], [], []
    exp.train_metrics = exp._init_metrics()
    exp.training_start_time = exp.training_start_time_corrected = time.time()

    exp.model.train()
    it = iter(_cycle(exp.train_loader))

    warm = int(OmegaConf.select(cfg, "prof.warm", default=WARM_STEPS))
    active = int(OmegaConf.select(cfg, "prof.active", default=ACTIVE))
    prof_memory = bool(OmegaConf.select(cfg, "prof.memory", default=False))

    # provenance: which posture is actually being profiled (the row's own knobs)
    knob = OmegaConf.select(cfg, "model.compile",
                            default=OmegaConf.select(cfg, "model.net.compile", default=None))
    gp_impl = OmegaConf.select(cfg, "model.net.gp_impl", default=None)
    device = torch.cuda.get_device_name(0) if cuda else "cpu"
    LOGGER.info(f"posture: batchsize={cfg.training.batchsize} compile={knob} "
                + (f"gp_impl={gp_impl} " if gp_impl else "") + f"device={device}")

    LOGGER.info(f"Warm-up: {warm} full training steps outside the profiler "
                f"(compilation + autotune land here, not in the tables)")
    for step in range(warm):
        exp._step(next(it), step)
    if cuda:
        torch.cuda.synchronize()
    try:  # recompile detector: a graph compiled INSIDE the window pollutes it
        graphs_before = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    except Exception:
        graphs_before = None

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    n_profiled = WAIT + WARMUP + active
    walls = []
    with profile(activities=activities,
                 schedule=schedule(wait=WAIT, warmup=WARMUP, active=active),
                 with_stack=True, profile_memory=prof_memory) as prof:
        for step in range(warm, warm + n_profiled):
            t0 = time.perf_counter()
            # no explicit synchronize here -- it would plant this tool's own sync in the
            # very tables it prints. _step's loss.item() already drains the queue each
            # step, so the wall is bounded, and a recompile/autotune spike is host-side
            # work that shows in the wall regardless.
            exp._step(next(it), step)
            walls.append(time.perf_counter() - t0)
            prof.step()
    if cuda:
        torch.cuda.synchronize()
    dt = sorted(walls)[len(walls) // 2]  # median profiled step

    # A window with a recompile or an autotune spike in it answers nothing. Say so
    # loudly instead of letting the tables be read as steady state.
    if graphs_before is not None:
        try:
            graphs_after = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            if graphs_after > graphs_before:
                LOGGER.warning(
                    f"{graphs_after - graphs_before} new dynamo graph(s) compiled INSIDE "
                    f"the profiled window -- the tables below are polluted; raise "
                    f"+prof.warm and re-run.")
        except Exception:
            pass
    if max(walls) > 2.5 * dt:
        LOGGER.warning(
            f"profiled step walls {['%.0f ms' % (w * 1e3) for w in walls]}: the outlier "
            f"suggests a recompile/autotune landed in-window; raise +prof.warm and re-run.")

    model_name = (OmegaConf.select(cfg, "model.net._target_", default="")
                  or "model").rsplit(".", 1)[-1]
    averages = prof.key_averages(group_by_stack_n=5)

    # ---- STEP 1a: the explicit stall list -------------------------------------------
    LOGGER.info("=" * 78)
    LOGGER.info(f"STEP 1 -- sync hunt ({model_name}, {active} steady-state steps, "
                f"median {dt * 1e3:.0f} ms/step wall)")
    stalls = [e for e in prof.key_averages()  # ungrouped: one row per event kind
              if any(k in e.key for k in STALL_KEYS)]
    if stalls:
        LOGGER.info(f"{'calls':>7}  {'self cpu total':>14}  event")
        for e in sorted(stalls, key=lambda e: -e.self_cpu_time_total):
            LOGGER.info(f"{e.count:7d}  {e.self_cpu_time_total / 1e3:11.2f} ms  {e.key}")
        LOGGER.info(
            "Known, not rediscoveries (fix shape = ONE host read per step): loss.item() "
            "+ the non-finite guard's isfinite every step; tracker .cpu().item()s on "
            "learned-frames rows. Anything BEYOND those counts is the finding.")
    else:
        LOGGER.info("no explicit stall events recorded"
                    + ("" if cuda else " (expected: no CUDA activity off-GPU)"))

    # ---- STEP 1b: CPU-side table with stacks (names the python line per stall) ------
    LOGGER.info(averages.table(sort_by="self_cpu_time_total", row_limit=30))

    # ---- STEP 1c: stall ATTRIBUTION -- the python line behind each sync/item site ----
    # The 1b table aggregates per op kind, which is how 353-1443 aten::item/step stayed
    # unattributed across two gate days: the counts were visible, the call sites were
    # not. group_by_stack_n rows carry the captured stack; print the top offenders with
    # their non-library frames so every stall names OUR line.
    attrib = [e for e in averages
              if any(k in e.key for k in ("cudaStreamSynchronize", "aten::item",
                                          "aten::_local_scalar_dense"))
              and e.self_cpu_time_total > 0]
    attrib.sort(key=lambda e: -e.self_cpu_time_total)
    if attrib:
        LOGGER.info("STEP 1c -- stall attribution (top sync/item sites by self-cpu; "
                    "frames filtered to this repo)")
        for e in attrib[:12]:
            frames = [f for f in (getattr(e, "stack", None) or [])
                      if "site-packages" not in f and "dist-packages" not in f]
            frames = frames[:3] or (getattr(e, "stack", None) or [])[:2]
            LOGGER.info(f"  {e.key}: {e.count} calls, "
                        f"{e.self_cpu_time_total / 1e3:.1f} ms self-cpu")
            for f in frames:
                LOGGER.info(f"      {f.strip()}")
        if not any(getattr(e, "stack", None) for e in attrib):
            LOGGER.info("  (no stacks captured on this torch build -- rerun with the "
                        "chrome trace and read the flow events instead)")

    # ---- STEP 2: hot kernels (the census shortlist: clone/permute marshalling, the
    # bmm-over-16-blades family, index/scatter backwards -- match them by name) -------
    if cuda:
        LOGGER.info("STEP 2 -- hot kernels (compare against the kernel-census shortlist)")
        LOGGER.info(prof.key_averages().table(sort_by="self_cuda_time_total",
                                              row_limit=25))

    os.makedirs("profiles", exist_ok=True)
    trace = os.path.join("profiles",
                         f"sync_{model_name}_bs{cfg.training.batchsize}.json.gz")
    prof.export_chrome_trace(trace)
    LOGGER.info(f"chrome trace: {os.path.abspath(trace)} -- read the GPU row's gaps "
                f"(drained queue between kernel spans = the launch-bound signature)")


if __name__ == "__main__":
    main()
