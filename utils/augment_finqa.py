#!/usr/bin/env python3
"""
Create a new dataset by linking FinQA entries with edgartools_finqa documents.

This script:
1. Loads FinQA dataset entries
2. For each entry, finds the matching document in edgartools_finqa
3. Uses text similarity to match documents when direct filename mapping fails
4. Creates a new dataset with linked document references
"""

import json
import os
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and punctuation."""
    # Convert to lowercase and remove extra whitespace
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    # Remove common formatting artifacts
    text = re.sub(r'[,\.\(\)\$%]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_numbers(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract numerical values from text, separating years, amounts, and other numbers."""
    # Find years (4-digit numbers that look like years)
    years = re.findall(r'\b(19|20)\d{2}\b', text)

    # Find currency amounts and large numbers (likely financial amounts)
    currency_amounts = re.findall(r'\$\s*\d+(?:[,.]\d+)*(?:\s*(?:million|billion|thousand))?', text, re.IGNORECASE)
    large_numbers = re.findall(r'\b\d+(?:[,.]\d+)*\s*(?:million|billion|thousand)\b', text, re.IGNORECASE)

    # Find other numbers including decimals, percentages, and scientific notation
    other_numbers = re.findall(r'\b\d+(?:[,.]\d+)*(?:[eE][+-]?\d+)?\b', text)
    percent_numbers = re.findall(r'\d+(?:\.\d+)?\s*%', text)

    # Normalize amounts
    normalized_amounts = []
    for amt in currency_amounts + large_numbers:
        clean_amt = re.sub(r'[\$,\s]', '', amt.lower())
        normalized_amounts.append(clean_amt)

    # Normalize other numbers
    normalized_others = []
    for num in other_numbers + percent_numbers:
        clean_num = re.sub(r'[\$%,\s]', '', num)
        if clean_num and clean_num not in years:  # Avoid duplicating years
            normalized_others.append(clean_num)

    return years, normalized_amounts, normalized_others


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts with heavy emphasis on years and financial amounts."""
    # Normalize texts
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)

    # Basic string similarity
    string_sim = SequenceMatcher(None, norm_text1, norm_text2).ratio()

    # Extract categorized numbers
    years1, amounts1, others1 = extract_numbers(text1)
    years2, amounts2, others2 = extract_numbers(text2)

    # Calculate individual similarity scores
    year_sim = 0
    amount_sim = 0
    other_sim = 0

    # Year matching (highest priority)
    if years1 and years2:
        common_years = len(set(years1).intersection(set(years2)))
        total_years = len(set(years1).union(set(years2)))
        year_sim = common_years / total_years if total_years > 0 else 0

    # Amount matching (second priority)
    if amounts1 and amounts2:
        common_amounts = len(set(amounts1).intersection(set(amounts2)))
        total_amounts = len(set(amounts1).union(set(amounts2)))
        amount_sim = common_amounts / total_amounts if total_amounts > 0 else 0

    # Other number matching (third priority)
    if others1 and others2:
        common_others = len(set(others1).intersection(set(others2)))
        total_others = len(set(others1).union(set(others2)))
        other_sim = common_others / total_others if total_others > 0 else 0

    # Weighted combination with heavy emphasis on years and amounts
    if year_sim > 0:
        # If years match, this is very likely the right document
        return min(1.0, 0.1 * string_sim + 0.6 * year_sim + 0.2 * amount_sim + 0.1 * other_sim + 0.3)  # Bonus for year match
    elif amount_sim > 0:
        # If amounts match but no years, still good but lower confidence
        return min(1.0, 0.2 * string_sim + 0.5 * amount_sim + 0.3 * other_sim)
    elif other_sim > 0:
        # Other numbers match
        return min(1.0, 0.4 * string_sim + 0.6 * other_sim)
    else:
        # No number matches, rely on string similarity but penalize
        return 0.5 * string_sim


def find_best_match_in_document(sentences: List[str], doc_content: str, threshold: float = 0.4) -> Tuple[float, List[str]]:
    """Find the best matching sentences in a document, focusing only on input sentences."""
    best_matches = []
    total_score = 0
    processed_sentences = 0

    # Split document into sentences/paragraphs (limit for performance)
    doc_lines = [line.strip() for line in doc_content.split('\n') if line.strip()][:1000]

    # Process only sentences with meaningful content and prioritize those with years/amounts
    valid_sentences = []
    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
        years, amounts, others = extract_numbers(sentence)
        # Prioritize sentences with years or financial amounts
        priority_score = len(years) * 3 + len(amounts) * 2 + len(others)
        valid_sentences.append((sentence, priority_score))

    # Sort by priority (sentences with years and amounts first)
    valid_sentences.sort(key=lambda x: x[1], reverse=True)

    # Process top sentences (focus on most numerical ones)
    for sentence, priority_score in valid_sentences[:8]:
        best_score = 0
        best_match = ""

        # Compare this specific sentence with each line in document
        for doc_line in doc_lines:
            if len(doc_line.strip()) < 10:
                continue

            score = calculate_similarity(sentence, doc_line)
            if score > best_score:
                best_score = score
                best_match = doc_line

        # Use lower threshold for high-priority sentences (with years/amounts)
        effective_threshold = threshold * 0.6 if priority_score > 2 else threshold

        if best_score > effective_threshold:
            best_matches.append(best_match)
            total_score += best_score
            processed_sentences += 1

    avg_score = total_score / processed_sentences if processed_sentences > 0 else 0
    return avg_score, best_matches


def find_matching_document(company: str, year: str, pre_text: List[str], post_text: List[str],
                         edgar_dir: Path) -> Tuple[Optional[str], float, List[str], str]:
    """Find the best matching document for a FinQA entry, comparing current and previous year scores."""

    # Define search order: same year first, then previous year
    current_year_file = edgar_dir / f"{company}_{year}.md"
    previous_year_file = edgar_dir / f"{company}_{int(year)-1}.md"

    # Combine pre_text and post_text for matching
    all_text = pre_text + post_text

    current_score = 0
    previous_score = 0
    current_matches = []
    previous_matches = []

    # Try same year document
    if current_year_file.exists():
        try:
            with open(current_year_file, 'r', encoding='utf-8') as f:
                doc_content = f.read()
            current_score, current_matches = find_best_match_in_document(all_text, doc_content)
        except Exception as e:
            pass  # Silently handle errors

    # Try previous year document
    if previous_year_file.exists():
        try:
            with open(previous_year_file, 'r', encoding='utf-8') as f:
                doc_content = f.read()
            previous_score, previous_matches = find_best_match_in_document(all_text, doc_content)
        except Exception as e:
            pass  # Silently handle errors

    # Compare scores and choose the best match
    # If there's a significant difference (>0.1), choose the higher score regardless of year
    if abs(current_score - previous_score) > 0.1:
        if current_score > previous_score and current_year_file.exists():
            return current_year_file.name, current_score, current_matches, "same_year"
        elif previous_score > current_score and previous_year_file.exists():
            return previous_year_file.name, previous_score, previous_matches, "previous_year"

    # If scores are close, prefer current year (as financial data often references previous periods)
    if current_score >= previous_score and current_year_file.exists():
        return current_year_file.name, current_score, current_matches, "same_year"
    elif previous_score > 0 and previous_year_file.exists():
        return previous_year_file.name, previous_score, previous_matches, "previous_year"

    # Fallback to any available document
    elif current_year_file.exists():
        return current_year_file.name, current_score, current_matches, "same_year"
    elif previous_year_file.exists():
        return previous_year_file.name, previous_score, previous_matches, "previous_year"

    return None, 0.0, [], "no_match"


def process_single_entry(entry: Dict, edgar_dir: Path, stats_lock: threading.Lock,
                        company_stats: Dict, match_stats: Dict) -> Dict:
    """Process a single FinQA entry."""
    # Extract company and year from filename
    filename = entry["filename"]  # e.g., "ADI/2009/page_49.pdf"
    parts = filename.split("/")
    company = parts[0]
    year = parts[1]

    # Find matching document
    matched_doc, confidence, matched_text, match_type = find_matching_document(
        company, year, entry["pre_text"], entry["post_text"], edgar_dir
    )

    # Thread-safe statistics update
    with stats_lock:
        match_stats["total"] += 1

        # Initialize company stats if not exists
        if company not in company_stats:
            company_stats[company] = {"same_year": 0, "previous_year": 0, "no_match": 0}

        # Update company statistics
        company_stats[company][match_type] += 1

        # Update match statistics
        if matched_doc and confidence > 0.3:
            match_stats["matched"] += 1
            if confidence < 0.6:
                match_stats["low_confidence"] += 1
        else:
            match_stats["no_match"] += 1

    # Create linked entry
    linked_entry = {
        "pre_text": entry["pre_text"],
        "post_text": entry["post_text"],
        "filename": entry["filename"],
        "qa": {
            "question": entry["qa"]["question"],
            "answer": entry["qa"]["answer"],
            "explanation": entry["qa"]["explanation"]
        }
    }

    # Add document linking information
    if matched_doc and confidence > 0.3:
        linked_entry["linked_document"] = matched_doc
        linked_entry["match_confidence"] = confidence
        linked_entry["matched_text_samples"] = matched_text[:3]  # Keep first 3 matches
    else:
        linked_entry["linked_document"] = None
        linked_entry["match_confidence"] = 0.0
        linked_entry["matched_text_samples"] = []

    return linked_entry


def process_finqa_dataset(finqa_path: Path, edgar_dir: Path, output_path: Path,
                         max_entries: Optional[int] = None, num_threads: int = 4):
    """Process the FinQA dataset and create linked dataset."""

    print(f"Loading FinQA dataset from {finqa_path}")
    with open(finqa_path, 'r', encoding='utf-8') as f:
        finqa_data = json.load(f)

    if max_entries:
        finqa_data = finqa_data[:max_entries]
        print(f"Processing first {max_entries} entries")

    linked_dataset = []
    match_stats = {"total": 0, "matched": 0, "no_match": 0, "low_confidence": 0}
    company_stats = {}
    stats_lock = threading.Lock()

    print(f"Processing {len(finqa_data)} entries using {num_threads} threads...")

    # Use ThreadPoolExecutor with tqdm progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(process_single_entry, entry, edgar_dir, stats_lock, company_stats, match_stats): entry
            for entry in finqa_data
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(finqa_data), desc="Processing entries") as pbar:
            for future in as_completed(future_to_entry):
                try:
                    linked_entry = future.result()
                    linked_dataset.append(linked_entry)
                except Exception as e:
                    print(f"Error processing entry: {e}")
                finally:
                    pbar.update(1)

    # Save the linked dataset
    print(f"Saving linked dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(linked_dataset, f, indent=2, ensure_ascii=False)

    # Print company statistics
    print("\n=== Company Statistics ===")
    for company in sorted(company_stats.keys()):
        stats = company_stats[company]
        total = stats["same_year"] + stats["previous_year"] + stats["no_match"]
        print(f"{company}: Same year: {stats['same_year']}, Previous year: {stats['previous_year']}, No match: {stats['no_match']} (Total: {total})")

    # Print final statistics
    print("\n=== Final Statistics ===")
    print(f"Total entries processed: {match_stats['total']}")
    print(f"Successfully matched: {match_stats['matched']} ({match_stats['matched']/match_stats['total']:.1%})")
    print(f"No match found: {match_stats['no_match']} ({match_stats['no_match']/match_stats['total']:.1%})")
    print(f"Low confidence matches: {match_stats['low_confidence']} ({match_stats['low_confidence']/match_stats['total']:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Create linked FinQA dataset with edgartools_finqa documents")
    parser.add_argument("--finqa-path", type=Path, default="FinQA/dataset/train.json",
                       help="Path to FinQA train.json file")
    parser.add_argument("--edgar-dir", type=Path, default="edgartools_finqa",
                       help="Directory containing edgartools_finqa .md files")
    parser.add_argument("--output", type=Path, default="linked_finqa_dataset.json",
                       help="Output path for the linked dataset")
    parser.add_argument("--max-entries", type=int, default=None,
                       help="Maximum number of entries to process (for testing)")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of threads to use for processing (default: 4)")

    args = parser.parse_args()

    # Validate paths
    if not args.finqa_path.exists():
        print(f"Error: FinQA dataset not found at {args.finqa_path}")
        return 1

    if not args.edgar_dir.exists():
        print(f"Error: Edgar directory not found at {args.edgar_dir}")
        return 1

    # Process the dataset
    process_finqa_dataset(args.finqa_path, args.edgar_dir, args.output, args.max_entries, args.threads)

    return 0


if __name__ == "__main__":
    exit(main())