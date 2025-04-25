import pandas as pd
import torch

from data.loader import CodeDataset
from src.analysis.naturalness import calculate_npr_score
from src.utils.auroc import calculate_auroc


def batch_naturalness_analysis(dataset: CodeDataset, model, tokenizer):
    """
    Analyze NPR naturalness scores for all code samples in the dataset.
    
    Args:
        dataset: CodeDataset object
        model: Pretrained language model
        tokenizer: Corresponding tokenizer
        
    Returns:
        DataFrame with NPR scores
    """
    results = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Human code
        npr_human = calculate_npr_score(sample['human_code'], model, tokenizer)
        results.append({
            'code_type': 'human',
            'solution_num': i % 3 + 1,
            'npr_score': npr_human
        })

        # AI code
        npr_ai = calculate_npr_score(sample['ai_code'], model, tokenizer)
        results.append({
            'code_type': 'ai',
            'solution_num': i % 3 + 1,
            'npr_score': npr_ai
        })

    df = pd.DataFrame(results)
    
    # Group and summarize
    summary = df.groupby(['code_type', 'solution_num']).agg({
        'npr_score': ['mean', 'std']
    })
    
    return df, summary

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Load tokenizer and model
model_name = "Salesforce/codegen-350M-mono"  # or any other causal model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Run NPR analysis
dataset = CodeDataset('data/dataset1.csv')

npr_detailed, npr_summary = batch_naturalness_analysis(dataset, model, tokenizer)

labels = (npr_detailed['code_type'] == 'ai').astype(int).tolist()
scores = npr_detailed['npr_score'].tolist()

# Compute AUROC
auroc_score = calculate_auroc(scores, labels)
print(f"\nAUROC Score (based on NPR): {auroc_score:.4f}")

# Save and display
npr_detailed.to_csv('npr_detailed_results.csv', index=False)
npr_summary.to_csv('npr_summary_statistics.csv')
print("\nNPR Summary Statistics:")
print(npr_summary)
