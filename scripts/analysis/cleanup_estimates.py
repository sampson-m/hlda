#!/usr/bin/env python3
"""
Cleanup script for the estimates directory.

This script helps clean up the massive estimates directory by:
1. Identifying duplicate files
2. Removing old/obsolete files
3. Organizing remaining files better
4. Creating a cleanup report
"""

import os
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(directory):
    """Find duplicate files in a directory."""
    hash_dict = defaultdict(list)
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.png', '.csv', '.pdf')):
                filepath = os.path.join(root, filename)
                try:
                    file_hash = get_file_hash(filepath)
                    hash_dict[file_hash].append(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    # Return only files with duplicates
    return {k: v for k, v in hash_dict.items() if len(v) > 1}


def analyze_estimates_directory(estimates_dir="estimates"):
    """Analyze the estimates directory structure."""
    estimates_path = Path(estimates_dir)
    
    if not estimates_path.exists():
        print(f"✗ Estimates directory {estimates_dir} not found!")
        return
    
    print(f"Analyzing {estimates_dir} directory...")
    print("=" * 60)
    
    # Count files by type
    file_counts = defaultdict(int)
    total_size = 0
    
    for filepath in estimates_path.rglob("*"):
        if filepath.is_file():
            file_counts[filepath.suffix] += 1
            total_size += filepath.stat().st_size
    
    print("File counts by type:")
    for ext, count in sorted(file_counts.items()):
        print(f"  {ext}: {count:,} files")
    
    print(f"\nTotal size: {total_size / (1024**3):.2f} GB")
    
    # Find duplicates
    print("\nFinding duplicate files...")
    duplicates = find_duplicates(estimates_dir)
    
    if duplicates:
        print(f"Found {len(duplicates)} sets of duplicate files:")
        duplicate_size = 0
        for file_hash, filepaths in duplicates.items():
            size = os.path.getsize(filepaths[0])
            duplicate_size += size * (len(filepaths) - 1)
            print(f"  {len(filepaths)} copies of {os.path.basename(filepaths[0])} ({size / 1024:.1f} KB each)")
        
        print(f"\nPotential space savings: {duplicate_size / (1024**3):.2f} GB")
    else:
        print("No duplicate files found.")
    
    return duplicates


def cleanup_duplicates(duplicates, dry_run=True):
    """Remove duplicate files, keeping the shortest path."""
    if not duplicates:
        print("No duplicates to clean up.")
        return
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning up duplicates...")
    
    removed_count = 0
    saved_space = 0
    
    for file_hash, filepaths in duplicates.items():
        # Sort by path length to keep the shortest (usually most organized)
        filepaths.sort(key=lambda x: len(x))
        
        # Keep the first file, remove the rest
        keep_file = filepaths[0]
        remove_files = filepaths[1:]
        
        print(f"\nKeeping: {keep_file}")
        for remove_file in remove_files:
            print(f"  Removing: {remove_file}")
            if not dry_run:
                try:
                    file_size = os.path.getsize(remove_file)
                    os.remove(remove_file)
                    removed_count += 1
                    saved_space += file_size
                except Exception as e:
                    print(f"    Error removing {remove_file}: {e}")
    
    if not dry_run:
        print(f"\n✓ Removed {removed_count} duplicate files")
        print(f"✓ Saved {saved_space / (1024**3):.2f} GB")
    else:
        print(f"\n[DRY RUN] Would remove {len([f for files in duplicates.values() for f in files[1:]])} duplicate files")


def create_cleanup_report(estimates_dir="estimates"):
    """Create a detailed cleanup report."""
    estimates_path = Path(estimates_dir)
    
    report = []
    report.append("# Estimates Directory Cleanup Report")
    report.append(f"Generated on: {Path().cwd()}")
    report.append("")
    
    # Directory structure
    report.append("## Directory Structure")
    for item in sorted(estimates_path.iterdir()):
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            report.append(f"- {item.name}/ ({size / (1024**3):.2f} GB)")
        else:
            size = item.stat().st_size
            report.append(f"- {item.name} ({size / 1024:.1f} KB)")
    
    report.append("")
    
    # File counts by type
    file_counts = defaultdict(int)
    for filepath in estimates_path.rglob("*"):
        if filepath.is_file():
            file_counts[filepath.suffix] += 1
    
    report.append("## File Counts by Type")
    for ext, count in sorted(file_counts.items()):
        report.append(f"- {ext}: {count:,} files")
    
    report.append("")
    
    # Recommendations
    report.append("## Cleanup Recommendations")
    report.append("")
    report.append("### 1. Remove Duplicate Files")
    report.append("- Use `python3 scripts/analysis/cleanup_estimates.py --find-duplicates`")
    report.append("- Review duplicates and remove unnecessary copies")
    report.append("")
    report.append("### 2. Archive Old Results")
    report.append("- Move old simulation results to `estimates/_archive/`")
    report.append("- Keep only the most recent results for each dataset")
    report.append("")
    report.append("### 3. Compress Large Files")
    report.append("- Consider compressing PNG files to reduce size")
    report.append("- Use lossless compression for plots")
    report.append("")
    report.append("### 4. Organize by Dataset")
    report.append("- Ensure each dataset has its own subdirectory")
    report.append("- Use consistent naming conventions")
    
    # Save report
    report_path = estimates_path / "cleanup_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Cleanup report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean up estimates directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze directory structure")
    parser.add_argument("--find-duplicates", action="store_true", help="Find duplicate files")
    parser.add_argument("--clean-duplicates", action="store_true", help="Remove duplicate files")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for cleanup operations")
    parser.add_argument("--report", action="store_true", help="Generate cleanup report")
    parser.add_argument("--estimates-dir", default="estimates", help="Estimates directory path")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_estimates_directory(args.estimates_dir)
    
    if args.find_duplicates:
        duplicates = find_duplicates(args.estimates_dir)
        if duplicates:
            print(f"\nFound {len(duplicates)} sets of duplicate files:")
            for file_hash, filepaths in duplicates.items():
                print(f"\n{len(filepaths)} copies of {os.path.basename(filepaths[0])}:")
                for filepath in filepaths:
                    print(f"  {filepath}")
        else:
            print("No duplicate files found.")
    
    if args.clean_duplicates:
        duplicates = find_duplicates(args.estimates_dir)
        cleanup_duplicates(duplicates, dry_run=args.dry_run)
    
    if args.report:
        create_cleanup_report(args.estimates_dir)
    
    if not any([args.analyze, args.find_duplicates, args.clean_duplicates, args.report]):
        # Default: analyze and create report
        analyze_estimates_directory(args.estimates_dir)
        create_cleanup_report(args.estimates_dir)


if __name__ == "__main__":
    main() 