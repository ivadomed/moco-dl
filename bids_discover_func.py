"""
Discover functional (BOLD) fMRI NIfTI files from a BIDS dataset.

Handles both layouts:
    sub-XX/func/sub-XX_task-..._bold.nii.gz
    sub-XX/ses-YY/func/sub-XX_ses-YY_task-..._bold.nii.gz

Usage:
    python bids_discover_func.py /path/to/bids_root
    python bids_discover_func.py /path/to/bids_root --task rest --out manifest.csv
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


BOLD_PATTERN = re.compile(r".*_bold\.nii(\.gz)?$")

SUB_RE = re.compile(r"sub-([a-zA-Z0-9]+)")
SES_RE = re.compile(r"ses-([a-zA-Z0-9]+)")
TASK_RE = re.compile(r"task-([a-zA-Z0-9]+)")
RUN_RE = re.compile(r"run-([a-zA-Z0-9]+)")


@dataclass
class BoldScan:
    subject: str
    session: Optional[str]
    task: Optional[str]
    run: Optional[str]
    path: str


def _extract_entity(pattern: re.Pattern, filename: str) -> Optional[str]:
    m = pattern.search(filename)
    return m.group(1) if m else None


def find_func_scans(bids_root: str, task: Optional[str] = None) -> list:
    """
    Walk a BIDS root and return all *_bold.nii[.gz] files under any func/ folder,
    whether or not a ses-* level is present.

    Robust to:
    - missing ses- folders (single-session datasets)
    - multiple sessions per subject
    - multiple runs/tasks per session
    - derivatives/ and sourcedata/ folders present alongside raw data (skipped)
    """
    bids_root = Path(bids_root)
    if not bids_root.is_dir():
        raise NotADirectoryError(f"BIDS root not found: {bids_root}")

    skip_dirs = {"derivatives", "sourcedata", "code", ".git", ".datalad"}
    scans = []

    for func_dir in bids_root.rglob("func"):
        if not func_dir.is_dir():
            continue
        # skip anything inside derivatives/sourcedata/etc.
        if skip_dirs.intersection(p.name for p in func_dir.parents):
            continue
        # only descend into func/ dirs that sit directly under a sub-*(/ses-*)? path
        if not any(p.name.startswith("sub-") for p in [func_dir.parent, func_dir.parent.parent]):
            continue

        for f in func_dir.iterdir():
            if not f.is_file() or not BOLD_PATTERN.match(f.name):
                continue

            subject = _extract_entity(SUB_RE, f.name)
            session = _extract_entity(SES_RE, f.name)
            task_id = _extract_entity(TASK_RE, f.name)
            run_id = _extract_entity(RUN_RE, f.name)

            if subject is None:
                # fall back to inferring subject from the directory path
                subject = _extract_entity(SUB_RE, str(func_dir))

            if task is not None and task_id != task:
                continue

            scans.append(BoldScan(
                subject=subject,
                session=session,
                task=task_id,
                run=run_id,
                path=str(f.resolve()),
            ))

    # sort for reproducibility: subject, session, task, run
    scans.sort(key=lambda s: (
        s.subject or "",
        s.session or "",
        s.task or "",
        s.run or "",
    ))
    return scans


def write_manifest(scans: list, out_path: str):
    fieldnames = ["subject", "session", "task", "run", "path"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scan in scans:
            writer.writerow(asdict(scan))


def main():
    parser = argparse.ArgumentParser(description="Discover BIDS functional BOLD scans.")
    parser.add_argument("bids_root", help="Path to the BIDS dataset root")
    parser.add_argument("--task", default=None, help="Only include scans matching this task label")
    parser.add_argument("--out", default=None, help="Write a CSV manifest to this path")
    args = parser.parse_args()

    scans = find_func_scans(args.bids_root, task=args.task)

    print(f"Found {len(scans)} functional scan(s).")
    subjects = sorted({s.subject for s in scans if s.subject})
    print(f"Subjects: {len(subjects)}")

    for s in scans:
        ses_str = f"ses-{s.session}" if s.session else "(no session)"
        task_str = f"task-{s.task}" if s.task else "(no task label)"
        run_str = f"run-{s.run}" if s.run else ""
        print(f"  sub-{s.subject} | {ses_str} | {task_str} {run_str} -> {s.path}")

    if args.out:
        write_manifest(scans, args.out)
        print(f"\nManifest written to {args.out}")


if __name__ == "__main__":
    main()
