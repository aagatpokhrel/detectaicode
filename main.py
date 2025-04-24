from transformers import AutoModelForCausalLM, AutoTokenizer
from detectcodegpt.src.detection.detect import DetectCodeGPT

# Load a code generation model
model_name = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize detector
detector = DetectCodeGPT(model=model, tokenizer=tokenizer)

# Example code to analyze
human_code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

machine_code = """
def factorial(n):
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    return 1 if n == 0 else n * factorial(n - 1)
"""

# Detect machine-generated code
is_machine, score = detector.detect(human_code)
print(f"Human code - Score: {score:.2f}, Is machine: {is_machine}")

is_machine, score = detector.detect(machine_code)
print(f"Machine code - Score: {score:.2f}, Is machine: {is_machine}")