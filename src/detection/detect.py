from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from ..analysis.logrank import calculate_log_rank

# from ..utils.parser import parse_code_tokens
from .perturbation import perturb_code


class DetectCodeGPT:
    def __init__(self, model, tokenizer, alpha=0.5, beta=0.5, lambda_spaces=3, lambda_newlines=2):
        """
        Initialize DetectCodeGPT detector.
        
        Args:
            model: The language model used for scoring
            alpha: Fraction of space locations to perturb
            beta: Fraction of newline locations to perturb
            lambda_spaces: Poisson parameter for space insertion
            lambda_newlines: Poisson parameter for newline insertion
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.lambda_spaces = lambda_spaces
        self.lambda_newlines = lambda_newlines

    def calculate_npr_score(self, code: str, model, tokenizer, num_perturbations=20):
        orig_log_rank = calculate_log_rank(code, model, tokenizer)

        perturbed_log_ranks = []
        for _ in range(num_perturbations):
            perturbed = perturb_code(
                code,
                alpha=self.alpha,
                beta=self.beta,
                lambda_spaces=self.lambda_spaces,
                lambda_newlines=self.lambda_newlines
            )
            score = calculate_log_rank(perturbed, model, tokenizer)
            perturbed_log_ranks.append(score)

        mean_perturbed = np.mean(perturbed_log_ranks)
        return mean_perturbed - orig_log_rank
    
    def detect(self, code: str, num_perturbations: int = 20, threshold: float = None) -> Tuple[bool, float]:
        """
        Detect if code is machine-generated.
        
        Args:
            code: The code snippet to analyze
            num_perturbations: Number of perturbations to generate
            threshold: Decision threshold (if None, returns score)
            
        Returns:
            Tuple of (is_machine_generated, detection_score)
        """
        
        # Compute detection score (difference between original and perturbed)
        detection_score = self.calculate_npr_score(code, self.model, self.tokenizer, num_perturbations)
        
        # Apply threshold if provided
        if threshold is not None:
            return (detection_score > threshold, detection_score)
        return (None, detection_score)