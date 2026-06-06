
    python find_lr.py -cp config -cn jctagging model=tag_transformer save=false
    python find_lr.py -cp config -cn toptagging model=tag_transformer save=false
    python find_lr.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd save=false
 
It reuses the experiment's own `_batch_loss`, optimizer, scaler and dataloader,
so the measured loss-vs-lr curve reflects the exact training setup: param groups,
`lr_factor_framesnet`, gradient clipping and amp are all honoured. The base
learning rate in `training.lr` is *ignored* during the sweep; only the relative
ratios between param groups (e.g. framesnet vs net) are preserved.
 
The test follows the Leslie-Smith / fastai recipe: exponentially ramp the lr over
a few hundred batches, record an EMA-smoothed training loss, stop early if the
loss diverges, then suggest an lr via the steepest-descent heuristic.
 
Pass `save=false` so no run directory is created. Tune the sweep on the CLI under
`lr_find.*` (these keys are added at runtime, so prefix with `+` is optional):
 
    +lr_find.start_lr=1e-7   lowest lr in the sweep                 (default 1e-7)
    +lr_find.end_lr=1e1      highest lr in the sweep                (default 1e1)
    +lr_find.num_iter=200    number of batches in the sweep         (default 200)
    +lr_find.beta=0.98       EMA factor for loss smoothing          (default 0.98)
    +lr_find.diverge=5.0     stop when smoothed_loss > diverge*best (default 5.0)
    +lr_find.skip_start=5    points dropped from the start          (default 5)
    +lr_find.skip_end=5      points dropped from the end            (default 5)
    +lr_find.output=lr_finder.png   plot path                       (default lr_finder.png)
"""
 
import os
 
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
 
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.amplitudes.experimentxl import AmplitudeXLExperiment
from experiments.eventgen.processes import ttbarExperiment
from experiments.logger import LOGGER
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.tagging.toptagxlexperiment import TopTagXLExperiment
 
CONSTRUCTORS = {
    "toptagging": TopTaggingExperiment,
    "toptaggingft": TopTaggingFineTuneExperiment,
    "toptagxl": TopTagXLExperiment,
    "jctagging": JetClassTaggingExperiment,
    "amplitudes": AmplitudeExperiment,
    "amplitudesxl": AmplitudeXLExperiment,
    "ttbar": ttbarExperiment,
}
 
DEFAULTS = dict(
    start_lr=1e-7,
    end_lr=1e1,
    num_iter=200,
    beta=0.98,
    diverge=5.0,
    skip_start=5,
    skip_end=5,
    output="lr_finder.png",
)
 
 
def build_experiment(cfg):
    """Construct and partially initialize an experiment (no scheduler, no training).
 
    Mirrors BaseExperiment.full_run() up to the point where the optimizer and
    scaler exist, which is all the range test needs. Matches the init sequence
    used in the equivariance tests.
    """
    try:
        constructor = CONSTRUCTORS[cfg.exp_type]
    except KeyError as err:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented") from err
 
    exp = constructor(cfg, rank=0, world_size=1)
    exp._init()  # device, logger, (run dir is skipped when save=false)
    exp.init_physics()  # wire data dims -> model dims
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp._init_optimizer()
    exp._init_scaler()
 
    exp.model.to(exp.device)
    return exp
 
 
def _cycle(loader):
    while True:
        yield from loader
 
 
def range_test(exp, start_lr, end_lr, num_iter, beta, diverge):
    """Exponentially ramp the lr and record the EMA-smoothed training loss."""
    optimizer, scaler = exp.optimizer, exp.scaler
    cfg_training = exp.cfg.training
 
    # preserve the relative lr ratios between param groups (net vs framesnet vs ...)
    base_lr0 = optimizer.param_groups[0]["lr"]
    base_ratios = [pg["lr"] / base_lr0 for pg in optimizer.param_groups]
    gamma = (end_lr / start_lr) ** (1.0 / max(1, num_iter - 1))
 
    lrs, losses = [], []
    avg_loss, best_loss = 0.0, float("inf")
 
    exp.model.train()
    iterator = iter(_cycle(exp.train_loader))
    log_every = max(1, num_iter // 20)
 
    for step in range(num_iter):
        lr = start_lr * gamma**step
        for pg, ratio in zip(optimizer.param_groups, base_ratios):
            pg["lr"] = lr * ratio
 
        data = next(iterator)
        loss, _ = exp._batch_loss(data)
 
        # update step, mirroring BaseExperiment._step (scaler + optional clipping)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg_training.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                exp.model.parameters(),
                cfg_training.clip_grad_norm,
                error_if_nonfinite=False,
            )
        scaler.step(optimizer)
        scaler.update()
 
        loss_value = loss.detach().item()
        if not np.isfinite(loss_value):
            LOGGER.warning(f"Non-finite loss at step {step} (lr={lr:.2e}); stopping.")
            break
 
        # bias-corrected EMA of the loss
        avg_loss = beta * avg_loss + (1.0 - beta) * loss_value
        smoothed = avg_loss / (1.0 - beta ** (step + 1))
        lrs.append(lr)
        losses.append(smoothed)
 
        if step == 0 or smoothed < best_loss:
            best_loss = smoothed
        if smoothed > diverge * best_loss:
            LOGGER.info(f"Loss diverged at step {step} (lr={lr:.2e}); stopping early.")
            break
 
        if step % log_every == 0:
            LOGGER.info(f"step {step:4d}  lr={lr:.2e}  smoothed_loss={smoothed:.4f}")
 
    return np.array(lrs), np.array(losses)
 
 
def suggest_lr(lrs, losses, skip_start, skip_end):
    """Steepest-descent heuristic: lr at the minimum gradient of loss vs log(lr)."""
    if len(lrs) <= skip_start + skip_end + 2:
        skip_start, skip_end = 0, 0
    lr_trim = lrs[skip_start : len(lrs) - skip_end]
    loss_trim = losses[skip_start : len(losses) - skip_end]
 
    gradients = np.gradient(loss_trim, np.log(lr_trim))
    steepest = float(lr_trim[int(np.argmin(gradients))])
    min_loss_lr = float(lr_trim[int(np.argmin(loss_trim))])
    return steepest, min_loss_lr, lr_trim, loss_trim
 
 
def make_plot(lrs, losses, steepest, min_loss_lr, output):
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lrs, losses, color="#1f77b4")
    ax.axvline(steepest, color="#d62728", ls="--", label=f"steepest: {steepest:.2e}")
    ax.axvline(
        min_loss_lr / 10.0,
        color="#2ca02c",
        ls=":",
        label=f"loss-min / 10: {min_loss_lr / 10.0:.2e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed training loss")
    ax.set_title("LR range test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    LOGGER.info(f"Saved LR-finder plot to {os.path.abspath(output)}")
 
 
@hydra.main(config_path="config_quick", config_name="toptagging", version_base=None)
def main(cfg):
    # LR finding is single-process and never trains / evaluates / saves a model
    cfg.train = False
    cfg.evaluate = False
    cfg.plot = False
 
    params = dict(DEFAULTS)
    lr_find = OmegaConf.select(cfg, "lr_find", default=None)
    if lr_find is not None:
        overrides = OmegaConf.to_container(lr_find, resolve=True)
        params.update({k: v for k, v in overrides.items() if v is not None})
 
    exp = build_experiment(cfg)
    LOGGER.info(
        f"Running LR range test: {params['start_lr']:.1e} -> {params['end_lr']:.1e} "
        f"over <= {params['num_iter']} batches (batchsize={cfg.training.batchsize})"
    )
 
    lrs, losses = range_test(
        exp,
        start_lr=params["start_lr"],
        end_lr=params["end_lr"],
        num_iter=params["num_iter"],
        beta=params["beta"],
        diverge=params["diverge"],
    )
    if len(lrs) < 3:
        LOGGER.error(
            "Not enough points collected before divergence; "
            "try a smaller start_lr, a larger diverge, or more num_iter."
        )
        return
 
    steepest, min_loss_lr, _, _ = suggest_lr(
        lrs, losses, skip_start=params["skip_start"], skip_end=params["skip_end"]
    )
    make_plot(lrs, losses, steepest, min_loss_lr, params["output"])
    np.savez(os.path.splitext(params["output"])[0] + ".npz", lr=lrs, loss=losses)
 
    LOGGER.info("=" * 64)
    LOGGER.info(f"Suggested lr (steepest descent): {steepest:.2e}")
    LOGGER.info(f"Suggested lr (loss-min / 10):    {min_loss_lr / 10.0:.2e}")
    LOGGER.info(f"  ->  reuse with:  training.lr={steepest:.2e}")
    LOGGER.info("=" * 64)
 
 
if __name__ == "__main__":
    main()
 
