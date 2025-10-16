#############################################################################
# Module: metrics_calculation_all.py                                        #
# Description: Calculate WER, CER, and BLEU for THREE model comparisons    #
# Author: Hanno Müller                                                      #
# Date: 2025-10-14                                                          #
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


def calculate_all_metrics_three_models(reference, large_v3, fine_tuned, new_model):
    """
    Calculate all metrics for four-way comparison (reference + 3 models).
    
    Args:
        reference (str): Original/ground truth text
        large_v3 (str): Whisper Large V3 hypothesis
        fine_tuned (str): Fine-tuned model hypothesis
        new_model (str): New model hypothesis
        
    Returns:
        dict: Dictionary with all metric comparisons
    """
    metrics = {}
    
    # WER calculations (6 comparisons)
    metrics['WER_large_v3_vs_original'] = calculate_wer(reference, large_v3)
    metrics['WER_fine_tuned_vs_original'] = calculate_wer(reference, fine_tuned)
    metrics['WER_new_model_vs_original'] = calculate_wer(reference, new_model)
    metrics['WER_large_v3_vs_fine_tuned'] = calculate_wer(large_v3, fine_tuned)
    metrics['WER_large_v3_vs_new_model'] = calculate_wer(large_v3, new_model)
    metrics['WER_fine_tuned_vs_new_model'] = calculate_wer(fine_tuned, new_model)
    
    # CER calculations (6 comparisons)
    metrics['CER_large_v3_vs_original'] = calculate_cer(reference, large_v3)
    metrics['CER_fine_tuned_vs_original'] = calculate_cer(reference, fine_tuned)
    metrics['CER_new_model_vs_original'] = calculate_cer(reference, new_model)
    metrics['CER_large_v3_vs_fine_tuned'] = calculate_cer(large_v3, fine_tuned)
    metrics['CER_large_v3_vs_new_model'] = calculate_cer(large_v3, new_model)
    metrics['CER_fine_tuned_vs_new_model'] = calculate_cer(fine_tuned, new_model)
    
    # BLEU calculations (6 comparisons)
    metrics['BLEU_large_v3_vs_original'] = calculate_bleu_score(reference, large_v3)
    metrics['BLEU_fine_tuned_vs_original'] = calculate_bleu_score(reference, fine_tuned)
    metrics['BLEU_new_model_vs_original'] = calculate_bleu_score(reference, new_model)
    metrics['BLEU_large_v3_vs_fine_tuned'] = calculate_bleu_score(large_v3, fine_tuned)
    metrics['BLEU_large_v3_vs_new_model'] = calculate_bleu_score(large_v3, new_model)
    metrics['BLEU_fine_tuned_vs_new_model'] = calculate_bleu_score(fine_tuned, new_model)
    
    return metrics


def calculate_metrics_for_segments_three_models(segments_data):
    """
    Calculate metrics for a list of segments with three models.
    
    Args:
        segments_data (list): List of dictionaries with transcription data
        
    Returns:
        list: List of dictionaries with added metrics
    """
    print(f"Calculating metrics for {len(segments_data)} segments (3 models)...")
    
    # Count empty intervals for reporting
    empty_count = 0
    
    for i, segment in enumerate(segments_data):
        # Check if this is an empty interval
        if segment.get('is_empty_interval', False):
            # Assign NULL (None) for all metrics on empty intervals
            # Empty intervals have no content, so metrics are not applicable
            null_metrics = {
                # WER metrics
                'WER_large_v3_vs_original': None,
                'WER_fine_tuned_vs_original': None,
                'WER_new_model_vs_original': None,
                'WER_large_v3_vs_fine_tuned': None,
                'WER_large_v3_vs_new_model': None,
                'WER_fine_tuned_vs_new_model': None,
                # CER metrics
                'CER_large_v3_vs_original': None,
                'CER_fine_tuned_vs_original': None,
                'CER_new_model_vs_original': None,
                'CER_large_v3_vs_fine_tuned': None,
                'CER_large_v3_vs_new_model': None,
                'CER_fine_tuned_vs_new_model': None,
                # BLEU metrics
                'BLEU_large_v3_vs_original': None,
                'BLEU_fine_tuned_vs_original': None,
                'BLEU_new_model_vs_original': None,
                'BLEU_large_v3_vs_fine_tuned': None,
                'BLEU_large_v3_vs_new_model': None,
                'BLEU_fine_tuned_vs_new_model': None
            }
            segment.update(null_metrics)
            empty_count += 1
        else:
            # Extract transcriptions for non-empty segments
            original = segment.get('transcript_original', '')
            large_v3 = segment.get('transcript_large_v3', '')
            fine_tuned = segment.get('transcript_fine_tuned', '')
            new_model = segment.get('transcript_new_model', '')
            
            # Calculate all metrics
            metrics = calculate_all_metrics_three_models(original, large_v3, fine_tuned, new_model)
            
            # Add metrics to segment
            segment.update(metrics)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(segments_data)} segments")
    
    print("Metrics calculation completed")
    print(f"  - Regular segments: {len(segments_data) - empty_count}")
    print(f"  - Empty intervals (default metrics): {empty_count}")
    return segments_data


def print_metrics_summary_three_models(segments_data):
    """Print summary statistics of calculated metrics for three models."""
    if not segments_data:
        return
    
    # Extract all metric values
    metric_names = [
        # WER metrics
        'WER_large_v3_vs_original',
        'WER_fine_tuned_vs_original',
        'WER_new_model_vs_original',
        'WER_large_v3_vs_fine_tuned',
        'WER_large_v3_vs_new_model',
        'WER_fine_tuned_vs_new_model',
        # CER metrics
        'CER_large_v3_vs_original',
        'CER_fine_tuned_vs_original',
        'CER_new_model_vs_original',
        'CER_large_v3_vs_fine_tuned',
        'CER_large_v3_vs_new_model',
        'CER_fine_tuned_vs_new_model',
        # BLEU metrics
        'BLEU_large_v3_vs_original',
        'BLEU_fine_tuned_vs_original',
        'BLEU_new_model_vs_original',
        'BLEU_large_v3_vs_fine_tuned',
        'BLEU_large_v3_vs_new_model',
        'BLEU_fine_tuned_vs_new_model'
    ]
    
    print(f"\nMETRICS SUMMARY (3 MODELS):")
    print("=" * 70)
    
    for metric_name in metric_names:
        values = [s.get(metric_name, 0) for s in segments_data if s.get(metric_name) is not None]
        if values:
            print(f"{metric_name:<40}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        else:
            print(f"{metric_name:<40}: No data")
    
    print("=" * 70)
