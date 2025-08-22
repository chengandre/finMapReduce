#!/usr/bin/env python3
"""
Script to count tokens in all markdown files within marker_financebench and edgartools_finqa directories.
Uses the num_tokens_from_string function from utils.py in the finMapReduce directory.
"""

import os
import sys
from pathlib import Path

# Add the finMapReduce directory to Python path to import utils
sys.path.append('finMapReduce')

from finMapReduce.utils import num_tokens_from_string

def count_tokens_in_directory(dir_path, dir_name):
    """
    Count tokens in all markdown files within a directory.
    Returns (total_tokens, total_files, token_counts_list)
    """
    directory = Path(dir_path)

    if not directory.exists():
        print(f"Directory {directory} does not exist!")
        return 0, 0

    total_tokens = 0
    total_files = 0
    token_counts = []  # List of token counts for each file

    print(f"Scanning {dir_name} for markdown files...")
    print("-" * 80)

    # For edgartools_finqa, files are directly in the directory
    if dir_name == "edgartools_finqa":
        for md_file in directory.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Count tokens using cl100k_base encoding
                token_count = num_tokens_from_string(content, "cl100k_base")

                # print(f"{md_file.name}: {token_count:,} tokens")

                total_tokens += token_count
                total_files += 1
                token_counts.append(token_count)

            except Exception as e:
                print(f"Error reading {md_file}: {e}")

    # For marker_financebench, walk through subdirectories
    else:
        for subdir in directory.iterdir():
            if subdir.is_dir():
                # Look for markdown files in each subdirectory
                for md_file in subdir.glob("*.md"):
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Count tokens using cl100k_base encoding
                        token_count = num_tokens_from_string(content, "cl100k_base")

                        # print(f"{md_file.relative_to(directory)}: {token_count:,} tokens")

                        total_tokens += token_count
                        total_files += 1
                        token_counts.append(token_count)

                    except Exception as e:
                        print(f"Error reading {md_file}: {e}")

    return total_tokens, total_files, token_counts

def get_all_token_counts():
    """
    Get list of token counts for all markdown files in both datasets.
    Returns list suitable for creating histograms.
    """
    directories = [
        ("marker_financebench", "marker_financebench"),
        ("edgartools_finqa", "edgartools_finqa")
    ]

    all_token_counts = []

    for dir_path, dir_name in directories:
        _, _, token_counts = count_tokens_in_directory(dir_path, dir_name)
        all_token_counts.extend(token_counts)

    return all_token_counts

def get_token_counts_by_dataset():
    """
    Get separate token count lists for each dataset.
    Returns (financebench_tokens, finqa_tokens) for stacked histograms.
    """
    _, _, financebench_tokens = count_tokens_in_directory("marker_financebench", "marker_financebench")
    _, _, finqa_tokens = count_tokens_in_directory("edgartools_finqa", "edgartools_finqa")

    return financebench_tokens, finqa_tokens

def count_tokens_in_markdown_files():
    """
    Count tokens in all markdown files within marker_financebench and edgartools_finqa directories.
    """
    directories = [
        ("marker_financebench", "marker_financebench"),
        ("edgartools_finqa", "edgartools_finqa")
    ]

    grand_total_tokens = 0
    grand_total_files = 0

    for dir_path, dir_name in directories:
        total_tokens, total_files, _ = count_tokens_in_directory(dir_path, dir_name)

        print("-" * 80)
        print(f"{dir_name} - Files processed: {total_files}")
        print(f"{dir_name} - Total tokens: {total_tokens:,}")

        if total_files > 0:
            avg_tokens = total_tokens / total_files
            print(f"{dir_name} - Average tokens per file: {avg_tokens:,.1f}")

        grand_total_tokens += total_tokens
        grand_total_files += total_files
        print()

    print("=" * 80)
    print(f"GRAND TOTAL - Files processed: {grand_total_files}")
    print(f"GRAND TOTAL - Total tokens: {grand_total_tokens:,}")

    if grand_total_files > 0:
        grand_avg_tokens = grand_total_tokens / grand_total_files
        print(f"GRAND TOTAL - Average tokens per file: {grand_avg_tokens:,.1f}")

if __name__ == "__main__":
    count_tokens_in_markdown_files()