#!/usr/bin/env python3
"""
Script to find and delete HuggingFace model folders.
Looks for directories containing .safetensors files or other typical model files.
"""

import os
import sys
import shutil
from pathlib import Path

def is_hf_model_folder(folder_path):
    """Check if a folder looks like a HuggingFace model folder."""
    path = Path(folder_path)
    
    # Check for .safetensors files
    if list(path.glob('*.safetensors')):
        return True
    
    # Check for other model files (at least 2 indicators)
    indicator_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    found = sum(1 for f in indicator_files if (path / f).exists())
    
    return found >= 2

def find_hf_model_folders(root_dir):
    """Find all HuggingFace model folders in the directory tree."""
    model_folders = []
    root_path = Path(root_dir).resolve()

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        sys.exit(1)

    if not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)

    # Never delete the agent's trained model — preserved at task/final_model
    # (a symlink) and task/experiments/exp_*/final_model (the real dir).
    # V3 (post-V2 pilot): V2 agents also wrote final_model artifacts at
    # task/experiments/exp_*/artifacts/final_model_<suffix>/ (e.g.
    # final_model_initial_sft, final_model_repair_sft). The original
    # protected-paths list only covered exp_*/final_model so those nested
    # artifacts were wiped. Extend the skip list to also cover anything
    # under experiments/exp_*/artifacts/ that looks like a final-model dir,
    # and (more conservatively) the artifacts/ dir itself.
    skip_paths = {(root_path / "final_model").resolve()}
    for exp_dir in root_path.glob("experiments/exp_*"):
        skip_paths.add((exp_dir / "final_model").resolve())
        # V2-style nested artifacts/<final_model*> dirs.
        artifacts_dir = exp_dir / "artifacts"
        if artifacts_dir.is_dir():
            skip_paths.add(artifacts_dir.resolve())
            for nested in artifacts_dir.glob("final_model*"):
                if nested.is_dir():
                    skip_paths.add(nested.resolve())
            # Anything else under artifacts/ — be conservative and protect
            # every immediate child dir so we don't wipe e.g. tokenizer
            # snapshots, LoRA adapters, intermediate checkpoints, etc.
            for child in artifacts_dir.iterdir():
                if child.is_dir():
                    skip_paths.add(child.resolve())

    def contains_protected_path(candidate):
        """Return the protected path if `candidate` is an ancestor of (or
        equal to) any protected dir, else None. Prevents the
        ancestor-bypass where e.g. experiments/exp_001/ matches the
        HF-model heuristic at its root and shutil.rmtree() would take the
        protected child along with it."""
        cand = Path(candidate).resolve()
        for p in skip_paths:
            try:
                p.relative_to(cand)
            except ValueError:
                continue
            return p
        return None

    for dirpath, dirnames, filenames in os.walk(root_path):
        if Path(dirpath).resolve() in skip_paths:
            dirnames.clear()
            continue
        if is_hf_model_folder(dirpath):
            blocked_by = contains_protected_path(dirpath)
            if blocked_by is not None:
                print(
                    f"Skipping {dirpath}: contains protected path "
                    f"{blocked_by} (would remove agent's final_model)"
                )
                # Keep walking into children so we can clean up unprotected
                # siblings of the protected dir (don't clear dirnames).
                continue
            model_folders.append(dirpath)
            # Don't traverse into model folders
            dirnames.clear()

    return model_folders

def main():
    if len(sys.argv) != 2:
        print("Usage: python delete_hf_models.py <directory>")
        sys.exit(1)
    
    search_dir = sys.argv[1]
    
    print(f"Searching for HuggingFace model folders in: {search_dir}")
    model_folders = find_hf_model_folders(search_dir)
    
    if not model_folders:
        print("No HuggingFace model folders found.")
        return
    
    print(f"\nFound {len(model_folders)} model folder(s):")
    for folder in model_folders:
        print(f"  - {folder}")
    
    for folder in model_folders:
        try:
            shutil.rmtree(folder)
            print(f"Deleted: {folder}")
        except Exception as e:
            print(f"Error deleting {folder}: {e}")
    print("\nDeletion complete!")

if __name__ == '__main__':
    main()