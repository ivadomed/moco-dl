#!/usr/bin/env python3
"""
Download dMRI and fMRI data for the moco-dl project.

Datasets (from the moco-dl README):
    dMRI:
        - spine-generic/data-multi-subject (GitHub, git-annex)
    fMRI:
        - OpenNeuro: ds004386 (v1.1.2)
        - OpenNeuro: ds004616 (v1.1.1)
        - OpenNeuro: ds005075 (v1.0.1)
        - OpenNeuro: ds006729 (v1.0.0)

Strategy
--------
Both sources expose data through git + git-annex (OpenNeuro mirrors every
dataset on github.com/OpenNeuroDatasets/<id>). We:
    1) git clone the repo (small, just metadata + symlinks),
    2) git checkout the pinned tag/version,
    3) git annex get only the files we actually need (dwi/ for dMRI,
       func/*bold* for fMRI).

This keeps the download small (no T1/T2/anat, no derivatives), reproducible
(pinned versions), and resumable (rerun the script and it will skip what's
already downloaded).

Requirements
------------
    - git
    - git-annex >= 8 (`git annex version`)
    - Python >= 3.8 (only stdlib used)

Usage
-----
    # Download everything
    python download_moco_data.py /path/to/output_dir

    # Only dMRI
    python download_moco_data.py /path/to/output_dir --modes dmri

    # Only fMRI, only first 5 subjects per dataset (useful for a smoke test)
    python download_moco_data.py /path/to/output_dir --modes fmri --max-subjects 5

    # See what would happen without downloading
    python download_moco_data.py /path/to/output_dir --dry-run

    # Skip a specific OpenNeuro dataset
    python download_moco_data.py /path/to/output_dir --skip ds006729

    # Also fetch anat (T2w) — not needed by moco-dl but handy for SCT QC
    python download_moco_data.py /path/to/output_dir --include-anat
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

# spine-generic dMRI dataset. Pin to a release tag for reproducibility.
# Tag list: https://github.com/spine-generic/data-multi-subject/tags
SPINE_GENERIC = {
    "name": "spine-generic-multi-subject",
    "url": "https://github.com/spine-generic/data-multi-subject.git",
    "tag": "r20250310",   # latest stable release at time of writing
    "modality": "dmri",
}

# OpenNeuro fMRI datasets. Versions match the moco-dl README.
# OpenNeuro mirrors every dataset on github.com/OpenNeuroDatasets/<id>,
# with git-annex backing the actual NIfTI files.
OPENNEURO_DATASETS = [
    {"id": "ds004386", "version": "1.1.2", "modality": "fmri"},
    {"id": "ds004616", "version": "1.1.1", "modality": "fmri"},
    {"id": "ds005075", "version": "1.0.1", "modality": "fmri"},
    {"id": "ds006729", "version": "1.0.0", "modality": "fmri"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, cwd=None, check=True, dry_run=False):
    """Run a shell command, streaming output. Returns the CompletedProcess."""
    pretty = " ".join(str(c) for c in cmd)
    location = f" (in {cwd})" if cwd else ""
    print(f"  $ {pretty}{location}")
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=cwd, check=check)


def check_dependencies():
    """Verify git and git-annex are installed."""
    missing = []
    for tool in ("git", "git-annex"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print("ERROR: missing required tool(s): " + ", ".join(missing), file=sys.stderr)
        print("\nInstallation hints:", file=sys.stderr)
        print("  Ubuntu/Debian : sudo apt install git git-annex", file=sys.stderr)
        print("  macOS (brew)  : brew install git git-annex", file=sys.stderr)
        print("  conda         : conda install -c conda-forge git-annex", file=sys.stderr)
        sys.exit(1)

    # Warn if git-annex is too old. moco-dl's source datasets need >= v8.
    try:
        out = subprocess.check_output(["git-annex", "version"], text=True)
        first = out.splitlines()[0]
        # Looks like: "git-annex version: 10.20230626"
        ver_str = first.split(":", 1)[1].strip().split(".")[0]
        major = int(ver_str)
        if major < 8:
            print(f"WARNING: git-annex {first} is older than v8; consider upgrading.",
                  file=sys.stderr)
    except Exception:
        pass  # version check is best-effort


def git_clone_or_update(url, dest: Path, tag: str | None, dry_run: bool):
    """
    Clone the repo if it doesn't exist; otherwise fetch and checkout the tag.
    Idempotent — safe to rerun.
    """
    if dest.exists() and (dest / ".git").exists():
        print(f"  -> repo already cloned at {dest}, fetching updates")
        run(["git", "fetch", "--tags", "--prune"], cwd=dest, dry_run=dry_run)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", url, str(dest)], dry_run=dry_run)

    if tag and not dry_run:
        # Detach onto the requested tag
        run(["git", "checkout", tag], cwd=dest)
        # Initialize annex (no-op if already initialized)
        run(["git", "annex", "init"], cwd=dest, check=False)


def annex_get(repo: Path, include_patterns: list[str], dry_run: bool,
              max_subjects: int | None = None):
    """
    Download files matching the given include patterns via git-annex.

    git-annex supports `--include='glob'` filters. We pass each pattern
    separately and rely on annex's own deduplication.

    If max_subjects is given, we restrict to the first N sub-* directories.
    """
    if not repo.exists():
        print(f"  -> repo {repo} does not exist, skipping annex get", file=sys.stderr)
        return

    # Determine the subject directories to descend into
    if max_subjects is not None:
        subjects = sorted(p for p in repo.iterdir() if p.is_dir() and p.name.startswith("sub-"))
        subjects = subjects[:max_subjects]
        if not subjects:
            print("  -> no sub-* directories found yet; falling back to whole repo")
            targets = [repo]
        else:
            print(f"  -> limiting to first {len(subjects)} subjects")
            targets = subjects
    else:
        targets = [repo]

    for tgt in targets:
        for pattern in include_patterns:
            # `git annex get --include=PATTERN .` from inside the target dir
            run(
                ["git", "annex", "get", f"--include={pattern}", "."],
                cwd=tgt,
                check=False,         # missing files for some subjects shouldn't abort
                dry_run=dry_run,
            )


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------

def download_spine_generic(out_root: Path, args):
    cfg = SPINE_GENERIC
    print(f"\n=== {cfg['name']} (dMRI, tag={cfg['tag']}) ===")
    repo = out_root / "dmri" / cfg["name"]
    git_clone_or_update(cfg["url"], repo, cfg["tag"], dry_run=args.dry_run)

    # Files we actually need for moco-dl/preprocessing.py:
    #   dwi.nii.gz  + bval/bvec/json sidecars
    patterns = ["*_dwi.nii.gz", "*_dwi.bval", "*_dwi.bvec", "*_dwi.json"]
    if args.include_anat:
        patterns += ["*_T2w.nii.gz", "*_T2w.json"]

    annex_get(repo, patterns, dry_run=args.dry_run, max_subjects=args.max_subjects)


def download_openneuro(out_root: Path, ds: dict, args):
    ds_id, version = ds["id"], ds["version"]
    print(f"\n=== OpenNeuro {ds_id} (fMRI, v{version}) ===")
    url = f"https://github.com/OpenNeuroDatasets/{ds_id}.git"
    repo = out_root / "fmri" / ds_id

    # OpenNeuro tags releases as the version string (e.g. "1.1.2")
    git_clone_or_update(url, repo, tag=version, dry_run=args.dry_run)

    patterns = ["*_bold.nii.gz", "*_bold.json"]
    if args.include_anat:
        patterns += ["*_T2w.nii.gz", "*_T2w.json", "*_T1w.nii.gz", "*_T1w.json"]

    annex_get(repo, patterns, dry_run=args.dry_run, max_subjects=args.max_subjects)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Download dMRI/fMRI datasets used by ivadomed/moco-dl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("output_dir", type=Path,
                   help="Root directory where datasets will be saved.")
    p.add_argument("--modes", nargs="+", choices=["dmri", "fmri"],
                   default=["dmri", "fmri"],
                   help="Which modalities to download (default: both).")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Dataset IDs to skip (e.g. 'ds006729 spine-generic-multi-subject').")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="Limit each dataset to the first N subjects (for testing).")
    p.add_argument("--include-anat", action="store_true",
                   help="Also download anatomical (T1w/T2w) images. Not required by moco-dl.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    return p.parse_args()


def main():
    args = parse_args()
    check_dependencies()

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_root}")
    if args.dry_run:
        print("(dry run — no files will be downloaded)")

    # ----- dMRI -----
    if "dmri" in args.modes and SPINE_GENERIC["name"] not in args.skip:
        download_spine_generic(out_root, args)
    elif "dmri" in args.modes:
        print(f"\nSkipping {SPINE_GENERIC['name']} (in --skip list)")

    # ----- fMRI -----
    if "fmri" in args.modes:
        for ds in OPENNEURO_DATASETS:
            if ds["id"] in args.skip:
                print(f"\nSkipping {ds['id']} (in --skip list)")
                continue
            download_openneuro(out_root, ds, args)

    print("\nAll done.")
    print(f"Data is organized as:")
    print(f"  {out_root}/dmri/spine-generic-multi-subject/sub-*/dwi/")
    print(f"  {out_root}/fmri/ds00*/sub-*/[ses-*/]func/")


if __name__ == "__main__":
    main()
