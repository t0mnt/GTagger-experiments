import numpy as np
import os, sys
import hashlib
import tarfile
import wget

# dataset sizes: toptagging 1.5G, event-generation 4.7G, JetClass ~190G (full)
BASE_URL = "https://www.thphys.uni-heidelberg.de/~plehn/data"
FILENAMES = {
    "toptagging": "toptagging_full.npz",
    "event-generation": "event_generation_ttbar.hdf5",
}
DATA_DIR = "data"

# JetClass (Pythia) -- https://zenodo.org/records/6619768. The repo's JetClass loader
# (experiments/tagging/jetclassexperiment.py) reads
#     <data.data_dir>/{train_100M,val_5M,test_20M}/<ClassName>_<NNN>.root
# and config/jctagging.yaml sets data.data_dir = data/JetClass/Pythia -- which is exactly
# the layout these official tars unpack to, so no post-processing or path edits are needed.
JETCLASS_BASE = "https://zenodo.org/record/6619768/files"
JETCLASS = {
    # split: (extract subdir under data/JetClass, [(tar filename, md5), ...])
    "train": (
        "Pythia/train_100M",
        [
            (f"JetClass_Pythia_train_100M_part{i}.tar", md5)
            for i, md5 in enumerate(
                [
                    "de4fd2dca2e68ab3c85d5cfd3bcc65c3",
                    "9722a359c5ef697bea0fbf79bf50f003",
                    "1e9f66cd1f915f9d10e90ae1d7761720",
                    "47348fc8985319fa4806da87500482fa",
                    "6b0ce16bd93b442a8d51914466990279",
                    "416e347512e716de51d392bee327b8e9",
                    "e9b9c1557b1b39bf0a16e4ab631ae451",
                    "5bfc6cb285ccb7680cefa9ac82ad1a2e",
                    "540c1a0d66dfad78d2b363c5740ccf86",
                    "668f40b3275167ff7104c48317c0ae2a",
                ]
            )
        ],
    ),
    "val": ("Pythia", [("JetClass_Pythia_val_5M.tar", "7235ccb577ed85023ea3ab4d5e6160cf")]),
    "test": ("Pythia", [("JetClass_Pythia_test_20M.tar", "64e5156d26d101adeb43b8388207d767")]),
}


def load(filename):
    url = os.path.join(BASE_URL, filename)
    print(f"Started to download {url}")
    target_path = os.path.join(DATA_DIR, filename)
    wget.download(url, out=target_path)
    print("")
    print(f"Successfully downloaded {target_path}")


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_jetclass(splits):
    """Download + verify + extract the JetClass (Pythia) tars for the given splits.

    Idempotent: a tar whose md5 already matches is not re-downloaded, and an already
    extracted tar (marked by a hidden ``.<tar>.extracted`` file) is skipped. The tars are
    large (~190 GB total) and can be deleted after extraction to reclaim disk.
    """
    base = os.path.join(DATA_DIR, "JetClass")
    for split in splits:
        subdir, files = JETCLASS[split]
        dest = os.path.join(base, subdir)
        os.makedirs(dest, exist_ok=True)
        for fname, md5 in files:
            tar_path = os.path.join(base, fname)
            marker = os.path.join(base, f".{fname}.extracted")
            if os.path.exists(marker):
                print(f"{fname} already extracted, skipping")
                continue
            url = f"{JETCLASS_BASE}/{fname}"
            if os.path.exists(tar_path) and _md5(tar_path) == md5:
                print(f"{fname} already downloaded (md5 ok)")
            else:
                if os.path.exists(tar_path):
                    os.remove(tar_path)  # partial/corrupt -> re-download
                print(f"Downloading {url}")
                wget.download(url, out=tar_path)
                print("")
                if _md5(tar_path) != md5:
                    raise RuntimeError(f"md5 mismatch for {fname}; delete it and retry")
            print(f"Extracting {fname} -> {dest}")
            with tarfile.open(tar_path) as tar:
                try:
                    tar.extractall(dest, filter="data")  # python >= 3.12 safe extraction
                except TypeError:
                    tar.extractall(dest)
            open(marker, "w").close()
            print(f"Extracted {fname}  (you may delete {tar_path} to reclaim disk)")
    print(f"JetClass ready under {base}/Pythia -- matches config/jctagging.yaml data.data_dir.")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python data/collect_data.py "
            "<toptagging | eventgen | jetclass [train|val|test|all]>"
        )
        sys.exit(1)
    dataset = sys.argv[1]

    # collect toptagging dataset
    # this is a npz version of the original dataset at https://zenodo.org/records/2603256
    filename = FILENAMES["toptagging"]
    if dataset == "toptagging":
        load(filename)

    # collect event generation dataset
    # this dataset is described in https://arxiv.org/abs/2411.00446
    filename = FILENAMES["event-generation"]
    if dataset == "eventgen":
        import h5py
        import hdf5plugin  # noqa: F401  (registers the hdf5 filters used by the file)

        load(filename)
        filename = os.path.join(DATA_DIR, filename)
        with h5py.File(filename, "r") as file:
            for njets in range(5):
                data = file[f"ttbar+{njets}jet"]
                target_path = os.path.join(DATA_DIR, f"ttbar_{njets}j.npy")
                np.save(target_path, data)
                print(f"Successfully created {target_path}")

    # collect the JetClass tagging dataset (https://zenodo.org/records/6619768)
    # second arg selects the split(s); default 'all'. Full download is ~190 GB.
    if dataset == "jetclass":
        arg = sys.argv[2] if len(sys.argv) > 2 else "all"
        splits = ["train", "val", "test"] if arg == "all" else [arg]
        unknown = [s for s in splits if s not in JETCLASS]
        if unknown:
            print(f"Unknown JetClass split(s) {unknown}; choose from train/val/test/all")
            sys.exit(1)
        collect_jetclass(splits)


if __name__ == "__main__":
    main()
