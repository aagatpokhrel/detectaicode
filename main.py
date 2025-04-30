from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from data.loader import CodeDataset
from src.detection.detect import DetectCodeGPT

# Load a code generation model
model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize detector
detector = DetectCodeGPT(model=model, tokenizer=tokenizer)

dataset = CodeDataset('data/dataset1.csv')

scores = []
labels = []

for item in dataset:
    # For human code (label 0)
    human_code = item['human_code']
    _, human_score = detector.detect(human_code)  # get detection score
    scores.append(human_score)
    labels.append(0)

    # For AI code (label 1)
    ai_code = item['ai_code']
    _, ai_score = detector.detect(ai_code)
    scores.append(ai_score)
    labels.append(1)

print (labels, scores)
