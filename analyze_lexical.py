# analyze.py
import os
from collections import defaultdict

import pandas as pd
from tree_sitter_language_pack import get_parser

from data.loader import CodeDataset
from src.analysis.lexical import analyze_lexical_diversity


def batch_analyze(dataset: CodeDataset) -> pd.DataFrame:
    """Analyze all code samples and return results dataframe."""
    results = []
    parser = get_parser('python')
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Human
        human_raw = analyze_lexical_diversity(sample['human_code'], parser)
        human_metrics = {
            "total_tokens": human_raw["total_tokens"],
            "zipf.slope": human_raw["zipf"]["slope"],
            "heaps.b": human_raw["heaps"]["b"],
            "heaps.k": human_raw["heaps"]["k"],
            "code_type": "human",
            "solution_num": i % 3 + 1
        }
        results.append(human_metrics)

        # AI
        ai_raw = analyze_lexical_diversity(sample['ai_code'], parser)
        ai_metrics = {
            "total_tokens": ai_raw["total_tokens"],
            "zipf.slope": ai_raw["zipf"]["slope"],
            "heaps.b": ai_raw["heaps"]["b"],
            "heaps.k": ai_raw["heaps"]["k"],
            "code_type": "ai",
            "solution_num": i % 3 + 1
        }
        results.append(ai_metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = df.groupby(['code_type', 'solution_num']).agg({
        'total_tokens': ['mean', 'std'],
        'zipf.slope': ['mean', 'std'],
        'heaps.b': ['mean', 'std'],
        'heaps.k': ['mean', 'std']
    })
    
    return df, summary

def save_results(df: pd.DataFrame, summary: pd.DataFrame):
    """Save results to CSV files."""
    df.to_csv('results/detailed_results.csv', index=False)
    summary.to_csv('results/summary_statistics.csv')
    print("Results saved to detailed_results.csv and summary_statistics.csv")

if __name__ == "__main__":
    # Load dataset
    dataset = CodeDataset('data/dataset1.csv')
    
    # Run analysis
    detailed_results, summary_stats = batch_analyze(dataset)
    
    # Save results
    save_results(detailed_results, summary_stats)
    
    # Print quick summary
    print("\nSummary Statistics:")
    print(summary_stats)