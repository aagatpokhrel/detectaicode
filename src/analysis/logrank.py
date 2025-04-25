import numpy as np
import torch


def calculate_log_rank(code: str, model, tokenizer) -> float:
    """
    Calculate log rank for the code.
    """
    inputs = tokenizer(code, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    # Probabilities and rank calculations
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    ranks = torch.zeros_like(input_ids)
    for i in range(input_ids.size(1)):
        token_id = input_ids[0, i]
        rank = (sorted_indices[0, i] == token_id).nonzero().item()
        ranks[0, i] = rank + 1  # 1-based rank

    log_ranks = torch.log(ranks.float())
    avg_log_rank = log_ranks.mean().item()

    return avg_log_rank
