#!/usr/bin/env python3
"""
Aggregate multiple prepared moco-dl datasets into a single unified one.

Use case
--------
You ran ``dataset_preparation.py`` on each OpenNeuro fMRI dataset
(ds004386, ds004616, ds005075, ds006729). Now you want to train one model
over all of them. This script:

  1) Symlinks every per-dataset .pt file into a unified tree, prefixing
     subject names with the dataset ID to avoid 'sub-01' collisions.
  2) Writes a merged ``dataset.json`` with paths rewritten to the new
     locations.

The result is a directory that ``train.py`` can use directly.

Usage
-----
    python aggregate_datasets.py \\
        --inputs   raw_data/fmri/ds004386/prepared/fmri_dataset \\
                   raw_data/fmri/ds004616/prepared/fmri_dataset \\
                   raw_data/fmri/ds005075/prepared/fmri_dataset \\
                   raw_data/fmri/ds006729/prepared/fmri_dataset \\
        --output   raw_data/fmri/all_fmri_prepared \\
        --mode     fmri
"""

import argparse
import json
import sys
from pathlib import Path


def _short_id(path: Path) -> str:
    """
    Make a short, unique identifier for a dataset directory.

    We walk up from the prepared/<mode>_dataset dir to find the original
    dataset name (e.g. 'ds004386' or 'spine-generic-multi-subject').
    """
    # path is e.g. .../fmri/ds004386/prepared/fmri_dataset
    parts = path.resolve().parts
    # The dataset name is typically two levels up from <mode>_dataset
    for cand in reversed(parts):
        if cand.startswith("ds") or "spine-generic" in cand:
            return cand
    # Fall back to the deepest non-generic name
    for cand in reversed(parts):
        if cand not in ("prepared", "fmri_dataset", "dmri_dataset"):
            return cand
    return path.name


def aggregate(inputs: list[Path], output: Path, mode: str) -> None:
    output.mkdir(parents=True, exist_ok=True)

    merged = {
        "schema_version": "moco-dl-v2",
        "mode": mode,
        "seed": "aggregated",
        "training": [],
        "validation": [],
        "testing": [],
    }
    source_summary = []

    for src in inputs:
        src = src.resolve()
        if not (src / "dataset.json").exists():
            print(f"  [skip] {src}: no dataset.json — did dataset_preparation.py run here?",
                  file=sys.stderr)
            continue

        ds_id = _short_id(src)
        with open(src / "dataset.json") as f:
            sub_ds = json.load(f)
        if sub_ds.get("mode") not in (None, mode):
            print(f"  [skip] {src}: mode={sub_ds.get('mode')!r} != {mode!r}",
                  file=sys.stderr)
            continue

        print(f"\n--- {ds_id} from {src} ---")
        counts = {"training": 0, "validation": 0, "testing": 0}

        for split in ("training", "validation", "testing"):
            for entry in sub_ds.get(split, []):
                # Entries for training/validation contain a 'data' key (.pt file).
                # Entries for testing have 'raw'/'fixed'/'mask' (NIfTI files).
                new_entry = {}
                for key, rel in entry.items():
                    abs_src = (src / rel).resolve()
                    if not abs_src.exists():
                        print(f"    [warn] missing: {abs_src}", file=sys.stderr)
                        new_entry = None
                        break

                    # Rebuild the relative path with a dataset-ID prefix
                    rel_path = Path(rel)
                    # rel_path looks like training/sub-XX_ses-YY/func/file.pt
                    # We rename the second component (sub-...) with the ds_id prefix
                    parts = list(rel_path.parts)
                    if len(parts) >= 2 and parts[1].startswith("sub-"):
                        parts[1] = f"{ds_id}_{parts[1]}"
                    new_rel = Path(*parts)

                    abs_dst = output / new_rel
                    abs_dst.parent.mkdir(parents=True, exist_ok=True)
                    if abs_dst.exists() or abs_dst.is_symlink():
                        # Idempotent: a previous run already linked this entry
                        pass
                    else:
                        abs_dst.symlink_to(abs_src)
                    new_entry[key] = str(new_rel)

                if new_entry is not None:
                    merged[split].append(new_entry)
                    counts[split] += 1

        print(f"    training={counts['training']}  "
              f"validation={counts['validation']}  "
              f"testing={counts['testing']}")
        source_summary.append((ds_id, counts))

    merged["sources"] = [
        {"id": sid, **counts} for sid, counts in source_summary
    ]

    out_json = output / "dataset.json"
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)

    total_t = len(merged["training"])
    total_v = len(merged["validation"])
    total_te = len(merged["testing"])
    print(f"\nWrote {out_json}")
    print(f"  training:   {total_t}")
    print(f"  validation: {total_v}")
    print(f"  testing:    {total_te}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", type=Path, required=True,
                   help="Per-dataset prepared dirs (each contains dataset.json).")
    p.add_argument("--output", type=Path, required=True,
                   help="Destination directory for the aggregated dataset.")
    p.add_argument("--mode", choices=["dmri", "fmri"], required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(args.inputs, args.output, args.mode)


if __name__ == "__main__":
    main()
