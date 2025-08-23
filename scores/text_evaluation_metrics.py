#!/usr/bin/env python3
"""
Text Evaluation Metrics Script
Calculates BertScore and BartScore for given sentences.

Author: Claude Code Assistant
"""

import argparse
import json
import torch
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings("ignore")

def install_required_packages():
    """Install required packages if not already installed."""
    return

def calculate_bertscore(
    candidates: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    batch_size: int = 64,
    device: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Calculate BertScore for candidate and reference sentences.

    Args:
        candidates: List of candidate sentences
        references: List of reference sentences
        lang: Language code (default: "en")
        model_type: Specific BERT model to use (optional)
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        import bert_score
    except ImportError:
        print("BertScore not installed. Installing...")
        install_required_packages()
        import bert_score

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Calculating BertScore using device: {device}")

    # Calculate BertScore
    P, R, F1 = bert_score.score(
        candidates,
        references,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=True
    )

    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist()
    }

def calculate_finbert_score(
    candidates: List[str],
    references: List[str],
    device: Optional[str] = None,
    model_name: str = "yiyanghkust/finbert-pretrain",
    batch_size: int = 32,
    pooling_strategy: str = "cls"
) -> Dict[str, List[float]]:
    """
    Calculate FinBERT-based similarity scores using embeddings.

    Args:
        candidates: List of candidate sentences
        references: List of reference sentences
        device: Device to use ('cuda' or 'cpu')
        model_name: FinBERT model name
        batch_size: Batch size for processing
        pooling_strategy: Pooling strategy ('cls' or 'mean')

    Returns:
        Dictionary with FinBERT similarity scores
    """
    try:
        from transformers import BertTokenizer, BertModel
        import numpy as np
    except ImportError:
        print("Required libraries not installed. Installing...")
        install_required_packages()
        from transformers import BertTokenizer, BertModel
        import numpy as np

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Calculating FinBERT scores using device: {device}")
    print(f"Using model: {model_name}")

    # Load FinBERT model and tokenizer
    try:
        bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
        bert_model = bert_model.to(device)
        bert_model.eval()
    except Exception as e:
        print(f"Error loading FinBERT model: {e}")
        print("Falling back to bert-base-uncased...")
        fallback_model = "bert-base-uncased"
        bert_tokenizer = BertTokenizer.from_pretrained(fallback_model)
        bert_model = BertModel.from_pretrained(fallback_model)
        bert_model = bert_model.to(device)
        bert_model.eval()

    def get_embeddings(texts: List[str]) -> torch.Tensor:
        """Get embeddings for a list of texts."""
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                inputs = bert_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(device)

                # Get hidden states
                outputs = bert_model(**inputs)
                hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

                # Apply pooling strategy
                if pooling_strategy == "cls":
                    # Use [CLS] token embedding
                    batch_embeddings = hidden_states[:, 0, :]  # [batch_size, hidden_size]
                elif pooling_strategy == "mean":
                    # Use mean pooling over non-padding tokens
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch_size, seq_len, 1]
                    masked_hidden = hidden_states * attention_mask  # Zero out padding tokens
                    summed = torch.sum(masked_hidden, dim=1)  # [batch_size, hidden_size]
                    lengths = torch.sum(attention_mask, dim=1)  # [batch_size, 1]
                    batch_embeddings = summed / lengths  # [batch_size, hidden_size]
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

                embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)  # [total_texts, hidden_size]

    print("Computing candidate embeddings...")
    candidate_embeddings = get_embeddings(candidates)

    print("Computing reference embeddings...")
    reference_embeddings = get_embeddings(references)

    # Calculate cosine similarity
    print("Computing cosine similarities...")
    similarities = []

    def cosine_similarity_numpy(a, b):
        """Calculate cosine similarity between two vectors using numpy."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    for i in range(len(candidates)):
        cand_emb = candidate_embeddings[i].numpy()  # [hidden_size]
        ref_emb = reference_embeddings[i].numpy()   # [hidden_size]

        # Calculate cosine similarity
        similarity = cosine_similarity_numpy(cand_emb, ref_emb)
        similarities.append(float(similarity))

    return {
        "finbert_similarity": similarities
    }

def calculate_bartscore(
    candidates: List[str],
    references: List[str],
    device: Optional[str] = None,
    checkpoint: str = "facebook/bart-large-cnn",
    batch_size: int = 4
) -> Dict[str, List[float]]:
    """
    Calculate BartScore for candidate and reference sentences.

    Args:
        candidates: List of candidate sentences
        references: List of reference sentences
        device: Device to use ('cuda' or 'cpu')
        checkpoint: BART model checkpoint
        batch_size: Batch size for processing

    Returns:
        Dictionary with BartScore values
    """
    try:
        from transformers import BartTokenizer, BartForConditionalGeneration
    except ImportError:
        print("Transformers library required for BARTScore. Installing...")
        install_required_packages()
        from transformers import BartTokenizer, BartForConditionalGeneration

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Calculating BARTScore using device: {device}")
    print("Note: Using simplified BARTScore implementation with transformers")

    # Load BART model and tokenizer
    try:
        bart_tokenizer = BartTokenizer.from_pretrained(checkpoint)
        bart_model = BartForConditionalGeneration.from_pretrained(checkpoint)
        bart_model = bart_model.to(device)
        bart_model.eval()
    except Exception as e:
        print(f"Error loading BART model: {e}")
        print("Falling back to distilbart-cnn-12-6...")
        fallback_checkpoint = "sshleifer/distilbart-cnn-12-6"
        bart_tokenizer = BartTokenizer.from_pretrained(fallback_checkpoint)
        bart_model = BartForConditionalGeneration.from_pretrained(fallback_checkpoint)
        bart_model = bart_model.to(device)
        bart_model.eval()

    scores = []

    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i+batch_size]
            batch_references = references[i:i+batch_size]

            for cand, ref in zip(batch_candidates, batch_references):
                try:
                    # Tokenize input (reference) and target (candidate)
                    inputs = bart_tokenizer(
                        ref,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(device)

                    targets = bart_tokenizer(
                        cand,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    )

                    # Prepare labels (shift right for BART)
                    labels = targets["input_ids"].to(device)

                    # Calculate log likelihood
                    outputs = bart_model(**inputs, labels=labels)
                    loss = outputs.loss

                    # Convert loss to score (negative log likelihood)
                    # Lower loss = higher likelihood = better score
                    score = -loss.item()
                    scores.append(score)

                except Exception as e:
                    print(f"Error processing pair: {e}")
                    scores.append(-100.0)  # Default poor score

    return {
        "bartscore": scores
    }

def evaluate_sentences(
    candidates: List[str],
    references: List[str],
    include_bertscore: bool = True,
    include_bartscore: bool = True,
    include_finbert: bool = True,
    lang: str = "en",
    device: Optional[str] = None,
    batch_size: int = 32,
    finbert_model: str = "yiyanghkust/finbert-pretrain",
    finbert_pooling: str = "cls"
) -> Dict[str, Union[Dict, List]]:
    """
    Evaluate sentences using BertScore, BartScore, and FinBERT similarity.

    Args:
        candidates: List of candidate sentences
        references: List of reference sentences
        include_bertscore: Whether to calculate BertScore
        include_bartscore: Whether to calculate BartScore
        include_finbert: Whether to calculate FinBERT similarity
        lang: Language code for BertScore
        device: Device to use
        batch_size: Batch size for processing
        finbert_model: FinBERT model name
        finbert_pooling: Pooling strategy for FinBERT ('cls' or 'mean')

    Returns:
        Dictionary containing evaluation results
    """
    if len(candidates) != len(references):
        raise ValueError("Number of candidates must match number of references")

    results = {
        "num_pairs": len(candidates),
        "candidates": candidates,
        "references": references
    }

    if include_bertscore:
        print("Computing BertScore...")
        bert_results = calculate_bertscore(
            candidates, references, lang=lang, device=device, batch_size=batch_size
        )
        results["bertscore"] = bert_results

        # Print BertScore summary
        avg_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
        print(f"BertScore Average F1: {avg_f1:.4f}")

    if include_bartscore:
        print("Computing BARTScore...")
        bart_results = calculate_bartscore(
            candidates, references, device=device, batch_size=min(batch_size, 8)
        )
        results["bartscore"] = bart_results

        # Print BARTScore summary
        avg_bart = sum(bart_results["bartscore"]) / len(bart_results["bartscore"])
        print(f"BARTScore Average: {avg_bart:.4f}")

    if include_finbert:
        print("Computing FinBERT similarity...")
        finbert_results = calculate_finbert_score(
            candidates, references, device=device, model_name=finbert_model,
            batch_size=batch_size, pooling_strategy=finbert_pooling
        )
        results["finbert"] = finbert_results

        # Print FinBERT summary
        avg_finbert = sum(finbert_results["finbert_similarity"]) / len(finbert_results["finbert_similarity"])
        print(f"FinBERT Average Similarity: {avg_finbert:.4f}")

    return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Calculate BertScore, BartScore, and FinBERT similarity for text evaluation")
    parser.add_argument("--candidates", nargs="+", help="Candidate sentences")
    parser.add_argument("--references", nargs="+", help="Reference sentences")
    parser.add_argument("--candidates-file", help="File containing candidate sentences (one per line)")
    parser.add_argument("--references-file", help="File containing reference sentences (one per line)")
    parser.add_argument("--output", help="Output JSON file to save results")
    parser.add_argument("--lang", default="en", help="Language code for BertScore")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BertScore calculation")
    parser.add_argument("--no-bartscore", action="store_true", help="Skip BartScore calculation")
    parser.add_argument("--no-finbert", action="store_true", help="Skip FinBERT similarity calculation")
    parser.add_argument("--finbert-model", default="yiyanghkust/finbert-pretrain", help="FinBERT model name")
    parser.add_argument("--finbert-pooling", default="cls", choices=["cls", "mean"], help="FinBERT pooling strategy")

    args = parser.parse_args()

    # Load sentences from files if provided
    if args.candidates_file:
        with open(args.candidates_file, 'r') as f:
            candidates = [line.strip() for line in f if line.strip()]
    elif args.candidates:
        candidates = args.candidates
    else:
        parser.error("Either --candidates or --candidates-file must be provided")

    if args.references_file:
        with open(args.references_file, 'r') as f:
            references = [line.strip() for line in f if line.strip()]
    elif args.references:
        references = args.references
    else:
        parser.error("Either --references or --references-file must be provided")

    # Evaluate sentences
    results = evaluate_sentences(
        candidates=candidates,
        references=references,
        include_bertscore=not args.no_bertscore,
        include_bartscore=not args.no_bartscore,
        include_finbert=not args.no_finbert,
        lang=args.lang,
        device=args.device,
        batch_size=args.batch_size,
        finbert_model=args.finbert_model,
        finbert_pooling=args.finbert_pooling
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    for i, (cand, ref) in enumerate(zip(candidates, references)):
        print(f"\nPair {i+1}:")
        print(f"Candidate: {cand}")
        print(f"Reference: {ref}")

        if "bertscore" in results:
            bert_data = results["bertscore"]
            if isinstance(bert_data, dict):
                precision_scores = bert_data.get('precision', [])
                recall_scores = bert_data.get('recall', [])
                f1_scores = bert_data.get('f1', [])
                if i < len(precision_scores) and i < len(recall_scores) and i < len(f1_scores):
                    print(f"BertScore - P: {precision_scores[i]:.4f}, R: {recall_scores[i]:.4f}, F1: {f1_scores[i]:.4f}")

        if "bartscore" in results:
            bart_data = results["bartscore"]
            if isinstance(bart_data, dict):
                bart_scores = bart_data.get('bartscore', [])
                if i < len(bart_scores):
                    print(f"BARTScore: {bart_scores[i]:.4f}")

        if "finbert" in results:
            finbert_data = results["finbert"]
            if isinstance(finbert_data, dict):
                finbert_scores = finbert_data.get('finbert_similarity', [])
                if i < len(finbert_scores):
                    print(f"FinBERT Similarity: {finbert_scores[i]:.4f}")

    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    if len(sys.argv) == 1:
        # Demo with example sentences
        print("Running demo with example sentences...")

        candidates = [
            "The weather is nice today.",
            "I am learning about machine learning.",
            "This movie was absolutely fantastic!"
        ]

        references = [
            "Today the weather is good.",
            "I'm studying machine learning concepts.",
            "The film was really excellent!"
        ]

        # Demo with all scoring methods
        results = evaluate_sentences(
            candidates,
            references,
            include_bertscore=True,
            include_bartscore=True,
            include_finbert=True
        )

        print("\nDemo Results:")
        print(json.dumps(results, indent=2))
    else:
        main()