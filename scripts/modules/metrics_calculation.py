#############################################################################
# Module: metrics_calculation.py                                            #
# Description: Calculate WER, CER, and BLEU metrics for all comparisons     #
# Author: Hanno Müller                                                      #
# Date: 2025-09-03                                                          #
#############################################################################

import re
import string
from collections import Counter
import numpy as np


def normalize_text_for_metrics(text):
    """
    Normalize text for fair metric calculation.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER).
    
    Args:
        reference (str): Reference text
        hypothesis (str): Hypothesis text
        
    Returns:
        float: WER score (0.0 = perfect, higher = worse)
    """
    # Normalize texts
    ref_words = normalize_text_for_metrics(reference).split()
    hyp_words = normalize_text_for_metrics(hypothesis).split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Dynamic programming for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(
                    d[i-1, j] + 1,    # deletion
                    d[i, j-1] + 1,    # insertion
                    d[i-1, j-1] + 1   # substitution
                )
    
    # WER is edit distance divided by reference length
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return float(wer)


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER).
    
    Args:
        reference (str): Reference text
        hypothesis (str): Hypothesis text
        
    Returns:
        float: CER score (0.0 = perfect, higher = worse)
    """
    # Normalize texts (but keep characters)
    ref_text = normalize_text_for_metrics(reference).replace(' ', '')
    hyp_text = normalize_text_for_metrics(hypothesis).replace(' ', '')
    
    if len(ref_text) == 0:
        return 0.0 if len(hyp_text) == 0 else 1.0
    
    # Dynamic programming for character-level edit distance
    d = np.zeros((len(ref_text) + 1, len(hyp_text) + 1))
    
    # Initialize first row and column
    for i in range(len(ref_text) + 1):
        d[i, 0] = i
    for j in range(len(hyp_text) + 1):
        d[0, j] = j
    
    # Fill the matrix
    for i in range(1, len(ref_text) + 1):
        for j in range(1, len(hyp_text) + 1):
            if ref_text[i-1] == hyp_text[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(
                    d[i-1, j] + 1,    # deletion
                    d[i, j-1] + 1,    # insertion
                    d[i-1, j-1] + 1   # substitution
                )
    
    # CER is edit distance divided by reference length
    cer = d[len(ref_text), len(hyp_text)] / len(ref_text)
    return float(cer)


def calculate_bleu_score(reference, hypothesis, n_gram=4):
    """
    Calculate BLEU score.
    
    Args:
        reference (str): Reference text
        hypothesis (str): Hypothesis text
        n_gram (int): Maximum n-gram order
        
    Returns:
        float: BLEU score (0.0 to 1.0, higher = better)
    """
    # Normalize texts
    ref_words = normalize_text_for_metrics(reference).split()
    hyp_words = normalize_text_for_metrics(hypothesis).split()
    
    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0
    
    # Brevity penalty
    if len(hyp_words) > len(ref_words):
        bp = 1.0
    else:
        bp = np.exp(1 - len(ref_words) / len(hyp_words))
    
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, min(n_gram + 1, len(hyp_words) + 1)):
        # Get n-grams
        ref_ngrams = []
        hyp_ngrams = []
        
        for i in range(len(ref_words) - n + 1):
            ref_ngrams.append(tuple(ref_words[i:i+n]))
        
        for i in range(len(hyp_words) - n + 1):
            hyp_ngrams.append(tuple(hyp_words[i:i+n]))
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        ref_counter = Counter(ref_ngrams)
        hyp_counter = Counter(hyp_ngrams)
        
        matches = 0
        for ngram, count in hyp_counter.items():
            matches += min(count, ref_counter.get(ngram, 0))
        
        precision = matches / len(hyp_ngrams)
        precisions.append(precision)
    
    if len(precisions) == 0 or any(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    geometric_mean = np.exp(np.mean(np.log(precisions)))
    
    # BLEU score
    bleu = bp * geometric_mean
    return float(bleu)


def calculate_all_metrics(reference, hypothesis1, hypothesis2):
    """
    Calculate all metrics for three-way comparison.
    
    Args:
        reference (str): Original/ground truth text
        hypothesis1 (str): First hypothesis (e.g., Large V3)
        hypothesis2 (str): Second hypothesis (e.g., fine-tuned)
        
    Returns:
        dict: Dictionary with all metric comparisons
    """
    metrics = {}
    
    # WER calculations
    metrics['WER_h1_vs_ref'] = calculate_wer(reference, hypothesis1)
    metrics['WER_h2_vs_ref'] = calculate_wer(reference, hypothesis2)
    metrics['WER_h1_vs_h2'] = calculate_wer(hypothesis1, hypothesis2)
    
    # CER calculations
    metrics['CER_h1_vs_ref'] = calculate_cer(reference, hypothesis1)
    metrics['CER_h2_vs_ref'] = calculate_cer(reference, hypothesis2)
    metrics['CER_h1_vs_h2'] = calculate_cer(hypothesis1, hypothesis2)
    
    # BLEU calculations
    metrics['BLEU_h1_vs_ref'] = calculate_bleu_score(reference, hypothesis1)
    metrics['BLEU_h2_vs_ref'] = calculate_bleu_score(reference, hypothesis2)
    metrics['BLEU_h1_vs_h2'] = calculate_bleu_score(hypothesis1, hypothesis2)
    
    return metrics


def calculate_metrics_for_segments(segments_data):
    """
    Calculate metrics for a list of segments.
    
    Args:
        segments_data (list): List of dictionaries with transcription data
        
    Returns:
        list: List of dictionaries with added metrics
    """
    print(f"Calculating metrics for {len(segments_data)} segments...")
    
    for i, segment in enumerate(segments_data):
        # Extract transcriptions
        original = segment.get('transcript_original', '')
        large_v3 = segment.get('transcript_large_v3', '')
        fine_tuned = segment.get('transcript_fine_tuned', '')
        
        # Calculate all metrics
        metrics = calculate_all_metrics(original, large_v3, fine_tuned)
        
        # Add metrics with descriptive names
        segment.update({
            # WER metrics
            'WER_large_v3_vs_original': metrics['WER_h1_vs_ref'],
            'WER_fine_tuned_vs_original': metrics['WER_h2_vs_ref'],
            'WER_large_v3_vs_fine_tuned': metrics['WER_h1_vs_h2'],
            
            # CER metrics
            'CER_large_v3_vs_original': metrics['CER_h1_vs_ref'],
            'CER_fine_tuned_vs_original': metrics['CER_h2_vs_ref'],
            'CER_large_v3_vs_fine_tuned': metrics['CER_h1_vs_h2'],
            
            # BLEU metrics
            'BLEU_large_v3_vs_original': metrics['BLEU_h1_vs_ref'],
            'BLEU_fine_tuned_vs_original': metrics['BLEU_h2_vs_ref'],
            'BLEU_large_v3_vs_fine_tuned': metrics['BLEU_h1_vs_h2'],
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(segments_data)} segments")
    
    print("Metrics calculation completed")
    return segments_data


def print_metrics_summary(segments_data):
    """Print summary statistics of calculated metrics."""
    if not segments_data:
        return
    
    # Extract all metric values
    metric_names = [
        'WER_large_v3_vs_original', 'WER_fine_tuned_vs_original', 'WER_large_v3_vs_fine_tuned',
        'CER_large_v3_vs_original', 'CER_fine_tuned_vs_original', 'CER_large_v3_vs_fine_tuned',
        'BLEU_large_v3_vs_original', 'BLEU_fine_tuned_vs_original', 'BLEU_large_v3_vs_fine_tuned'
    ]
    
    print(f"\nMETRICS SUMMARY:")
    print("=" * 60)
    
    for metric_name in metric_names:
        values = [s.get(metric_name, 0) for s in segments_data if s.get(metric_name) is not None]
        if values:
            print(f"{metric_name:<35}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        else:
            print(f"{metric_name:<35}: No data")
    
    print("=" * 60)
