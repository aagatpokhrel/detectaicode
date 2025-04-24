import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from .detection.detect import DetectCodeGPT
from .utils.metrics import calculate_auroc
from .utils.visualize import plot_detection_results

def main():
    parser = argparse.ArgumentParser(description="DetectCodeGPT: Machine-Generated Code Detection")
    parser.add_argument("--human_dir", type=str, required=True, help="Directory with human-written code")
    parser.add_argument("--machine_dir", type=str, required=True, help="Directory with machine-generated code")
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-hf", 
                        help="HuggingFace model name for detection")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Initialize detector
    detector = DetectCodeGPT(model=model, tokenizer=tokenizer)
    
    # Load and evaluate samples
    human_scores = evaluate_samples(Path(args.human_dir), detector, args.num_samples)
    machine_scores = evaluate_samples(Path(args.machine_dir), detector, args.num_samples)
    
    # Calculate AUROC
    auroc = calculate_auroc(human_scores, machine_scores)
    print(f"Detection AUROC: {auroc:.4f}")
    
    # Save and plot results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plot_detection_results(human_scores, machine_scores, output_dir / "detection_results.png")
    
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write("Human scores:\n")
        f.write("\n".join(map(str, human_scores)) + "\n")
        f.write("Machine scores:\n")
        f.write("\n".join(map(str, machine_scores)) + "\n")

def evaluate_samples(directory: Path, detector, num_samples: int):
    """Evaluate samples from directory and return detection scores."""
    scores = []
    sample_files = list(directory.glob("*.py"))[:num_samples]
    
    for i, file_path in enumerate(sample_files):
        with open(file_path, "r") as f:
            code = f.read()
        
        # Get detection score (higher = more likely machine-generated)
        _, score = detector.detect(code)
        scores.append(score)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(sample_files)} samples from {directory.name}")
    
    return scores

if __name__ == "__main__":
    main()