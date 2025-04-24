import numpy as np
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def calculate_npr_score(code: str, model, tokenizer) -> float:
    """
    Calculate Normalized Perturbed Log Rank (NPR) score for code.
    
    Args:
        code: Input code string
        model: Language model for scoring
        tokenizer: Corresponding tokenizer
        
    Returns:
        NPR score (higher means more likely machine-generated)
    """
    # Tokenize code and get model predictions
    inputs = tokenizer(code, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
    
    # Calculate log probabilities and ranks
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # Get ranks of actual tokens
    ranks = torch.zeros_like(input_ids)
    for i in range(input_ids.size(1)):
        token_id = input_ids[0, i]
        rank = (sorted_indices[0, i] == token_id).nonzero().item()
        ranks[0, i] = rank + 1  # 1-based ranking
    
    # Calculate log rank scores
    log_ranks = torch.log(ranks.float())
    avg_log_rank = log_ranks.mean().item()
    
    return avg_log_rank