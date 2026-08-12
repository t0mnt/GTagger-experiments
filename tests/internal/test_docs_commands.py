"""Every command the docs tell a reader to paste must be runnable as written.

Documentation rots silently: a renamed script or a moved config leaves a command that
looks right and fails only when someone pastes it into a cluster shell, often hours
into a session. These checks are static and cheap, and cover the failure modes seen
in practice -- unparseable blocks, references to scripts that moved (find_lr.py ->
utils/find_lr.py), and hydra overrides naming configs that do not exist.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = ["GUIDE.md", "docs/OSCAR.md", "docs/SLURM.md", "REPRODUCE.md"]


def _blocks(doc):
    text = (REPO / doc).read_text()
    raw = re.findall(r"^[ ]*```bash\n(.*?)^[ ]*```", text, re.M | re.S)
    return [re.sub(r"^  ", "", b, flags=re.M) for b in raw]


@pytest.mark.parametrize("doc", DOCS)
def test_bash_blocks_parse(doc):
    """A block that does not parse cannot be pasted -- e.g. a raw <placeholder>, which
    bash reads as a redirect."""
    for i, block in enumerate(_blocks(doc), 1):
        proc = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
        assert proc.returncode == 0, f"{doc} block {i} is not valid bash:\n{proc.stderr}\n{block[:400]}"


@pytest.mark.parametrize("doc", DOCS)
def test_referenced_scripts_exist(doc):
    """`python <path>` in the docs must point at a file that is actually there."""
    text = (REPO / doc).read_text()
    for script in set(re.findall(r"python3? ([\w/]+\.py)", text)):
        assert (REPO / script).is_file(), (
            f"{doc} references `python {script}`, which does not exist. If it moved, update "
            f"the docs (this is how the find_lr.py -> utils/find_lr.py move could have rotted)."
        )


@pytest.mark.parametrize("doc", DOCS)
def test_referenced_configs_exist(doc):
    """`model=X` / `training=Y` / `-cn Z` must name configs that compose."""
    text = (REPO / doc).read_text()
    missing = []

    def concrete(name):
        # docs legitimately write placeholders (`model=tag_<hybrid>`); the regex stops at
        # "<", leaving a trailing "_", so anything ending in _ is a placeholder stem
        return name and not name.endswith("_")

    for name in set(re.findall(r"\bmodel=([\w.\-]+)", text)):
        if name.startswith("tag_") and concrete(name) and not list(REPO.glob(f"config*/model/{name}.yaml")):
            missing.append(f"model={name}")
    for name in set(re.findall(r"\btraining=([\w.\-]+)", text)):
        if concrete(name) and not list(REPO.glob(f"config*/training/{name}.yaml")):
            missing.append(f"training={name}")
    for name in set(re.findall(r"-cn ([\w.\-]+)", text)):
        # `-cn config` names the config.yaml saved inside a run dir (warm starts), not a
        # file in the config*/ trees
        if concrete(name) and name != "config" and not list(REPO.glob(f"config*/{name}.yaml")):
            missing.append(f"-cn {name}")
    assert not missing, f"{doc} names configs that do not exist: {sorted(missing)}"


def test_sbatch_template_shifts_are_arity_guarded():
    """`shift 2` with one argument left fails WITHOUT shifting -> infinite loop.

    That is not hypothetical: a trailing `-cp` once spun the parser at 100% CPU holding
    a GPU for the job's full walltime. Any `shift 2` must be preceded by an arity check
    on the same line.
    """
    for line in (REPO / "docs" / "oscar-train.sbatch").read_text().splitlines():
        if "shift 2" in line:
            assert "$# -ge 2" in line, (
                f"unguarded `shift 2` in docs/oscar-train.sbatch:\n  {line.strip()}\n"
                f"With one argument left this fails without shifting and the while-loop "
                f"never terminates."
            )


def test_sbatch_template_parses():
    proc = subprocess.run(["bash", "-n", str(REPO / "docs" / "oscar-train.sbatch")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_sbatch_targets_are_created_before_first_use():
    """A file you are told to `sbatch` must be created earlier in the same document.

    `train.sbatch` is not in the repo -- it is a working copy the reader makes from
    `docs/oscar-train.sbatch`, gitignored on purpose so a partition or account can never reach
    this public repo. The copy step is therefore a PREREQUISITE and must appear above the first
    submission. It did not: OSCAR section 5 opened with the ParticleNet reproduction
    `sbatch ... train.sbatch ...` and put `cp docs/oscar-train.sbatch train.sbatch` (and
    `mkdir -p logs`) thirty lines later, so following the document top-to-bottom produced
    `sbatch: error: Unable to open file train.sbatch`.

    The rule: before the first submission, the filename must appear on some earlier line that is
    NOT itself an sbatch invocation -- i.e. somewhere it is being created or described, whether by
    `cp` (OSCAR) or by prose plus a file block ("save the following as the FILE ...", SLURM).

    `test_referenced_scripts_exist` does not cover this: it only checks `python <path>.py`.
    """
    for doc in DOCS:
        text = (REPO / doc).read_text()
        for m in re.finditer(r"^[ ]*sbatch\b[^\n]*?(\S+\.sbatch)\b", text, re.M):
            target, at = m.group(1), m.start()
            if (REPO / target).is_file():
                continue  # shipped in the repo; nothing for the reader to create
            earlier = [
                ln for ln in text[:at].splitlines() if target in ln and "sbatch " not in ln
            ]
            assert earlier, (
                f"{doc} line {text[:at].count(chr(10)) + 1}: `sbatch ... {target}` names a file "
                f"that is not in the repo and is never created earlier in the document. Move the "
                f"step that creates it above the first submission -- a reader pasting in order "
                f"gets 'Unable to open file'."
            )


# Widen the sweep past the four docs above: a broken command is just as costly when it
# lives in a config comment or a module docstring, and that is where this one lived.
SOURCES = DOCS + [
    "utils/find_lr.py", "utils/bperf.py", "docs/cgenn-compile.md",
    "config/model/tag_cgenn.yaml", "todo.md",
]


@pytest.mark.parametrize("src", SOURCES)
def test_hydra_config_path_is_valid_for_the_script(src):
    """`-cp X` is resolved by hydra RELATIVE TO THE SCRIPT, not the working directory.

    `python utils/find_lr.py -cp config ...` therefore looks for `utils/config` and dies
    with "Primary config directory not found" -- while the identical-looking
    `python run.py -cp config ...` is correct, because run.py sits at the repo root. Six
    such invocations shipped: four in find_lr.py's own usage header, one in
    docs/cgenn-compile.md, and one in the tag_cgenn.yaml comment giving the command that
    settles the gp_impl posture. find_lr.py already defaults to `../config`, so the fix is
    to drop `-cp` entirely.

    Static check, and it has to be: running each command costs minutes and a GPU.
    """
    text = (REPO / src).read_text()
    bad = []
    for script, cp in re.findall(r"python3? ([\w/]+\.py)\s+-cp\s+([\w./]+)", text):
        script_dir = (REPO / script).parent
        if not (script_dir / cp).is_dir():
            bad.append(f"`python {script} -cp {cp}` -> {script_dir / cp} does not exist")
    assert not bad, (
        f"{src} documents a config path that hydra resolves relative to the SCRIPT:\n  "
        + "\n  ".join(bad)
        + f"\nDrop `-cp` if the script's own hydra.main already points at the right tree."
    )
