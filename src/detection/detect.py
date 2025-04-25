from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from ..analysis.naturalness import calculate_npr_score

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
    
    def detect(self, code: str, num_perturbations: int = 50, threshold: float = None) -> Tuple[bool, float]:
        """
        Detect if code is machine-generated.
        
        Args:
            code: The code snippet to analyze
            num_perturbations: Number of perturbations to generate
            threshold: Decision threshold (if None, returns score)
            
        Returns:
            Tuple of (is_machine_generated, detection_score)
        """
        # Calculate original NPR score
        orig_score = calculate_npr_score(code, self.model, self.tokenizer)
        
        # Generate perturbations and calculate their scores
        perturbed_scores = []
        for _ in range(num_perturbations):
            perturbed_code = perturb_code(
                code,
                alpha=self.alpha,
                beta=self.beta,
                lambda_spaces=self.lambda_spaces,
                lambda_newlines=self.lambda_newlines
            )
            perturbed_score = calculate_npr_score(perturbed_code, self.model, self.tokenizer)
            perturbed_scores.append(perturbed_score)
        
        # Calculate mean perturbed score
        mean_perturbed_score = np.mean(perturbed_scores)
        
        # Compute detection score (difference between original and perturbed)
        detection_score = orig_score - mean_perturbed_score
        
        # Apply threshold if provided
        if threshold is not None:
            return (detection_score > threshold, detection_score)
        return (None, detection_score)