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
