import pandas as pd
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 1. Load model ────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str):
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

print("Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer("google/gemma-3-27b-it")

# ── 2. Load dataset ──────────────────────────────────────────────────────────
ecqa_data = pd.read_parquet("/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/datasets/ecqa/data/validation-00000-of-00001.parquet")
print(f"Dataset size: {len(ecqa_data)} questions")

# ── 3. Build prompt ──────────────────────────────────────────────────────────
OPTION_LABELS = ["A", "B", "C", "D", "E"]

def build_prompt(row):
    options_text = "\n".join(
        f"{OPTION_LABELS[i]}. {row[f'q_op{i+1}']}" for i in range(5)
    )
    return (
        f"Question: {row['q_text']}\n\n"
        f"{options_text}\n\n"
        f"Answer with ONLY the letter (A, B, C, D, or E) of the correct option."
    )

# ── 4. Extract predicted letter from model output ────────────────────────────
def extract_answer_letter(text: str) -> str | None:
    """Return the first A–E letter found in the model's response."""
    text = text.strip()
    # Try exact single-letter match first
    if text.upper() in OPTION_LABELS:
        return text.upper()
    # Look for patterns like "A.", "(A)", "Answer: A", etc.
    match = re.search(r'\b([A-E])\b', text.upper())
    return match.group(1) if match else None

# ── 5. Get the correct option's letter ───────────────────────────────────────
def get_correct_letter(row):
    """Match q_ans text to one of q_op1–q_op5 and return the letter."""
    answer = row["q_ans"].strip().lower()
    for i in range(5):
        if row[f"q_op{i+1}"].strip().lower() == answer:
            return OPTION_LABELS[i]
    return None  # shouldn't happen if data is clean

# ── 6. Run evaluation ────────────────────────────────────────────────────────
results = []
correct = 0
total = 0
extraction_failures = 0

for idx, row in tqdm(ecqa_data.iterrows(), total=len(ecqa_data), desc="Evaluating"):
    prompt = build_prompt(row)
    correct_letter = get_correct_letter(row)

    # Tokenize using chat template
    # Tokenize using chat template
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    predicted_letter = extract_answer_letter(response)
    if predicted_letter is None:
        extraction_failures += 1
        is_correct = False  # count as incorrect if we can't extract an answer
    else:
        is_correct = predicted_letter == correct_letter
    if is_correct:
        correct += 1
    total += 1

    results.append({
        "index": idx,
        "question": row["q_text"],
        "correct_answer": row["q_ans"],
        "correct_letter": correct_letter,
        "model_response": response.strip(),
        "predicted_letter": predicted_letter,
        "is_correct": is_correct,
    })

    # Print progress every 100 questions
    if total % 100 == 0:
        print(f"  [{total}/{len(ecqa_data)}] Running accuracy: {correct/total:.4f}")

# ── 7. Compute and save results ──────────────────────────────────────────────
accuracy = correct / total if total > 0 else 0.0
print(f"\n{'='*50}")
print(f"Final Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")

output = {
    "model": "google/gemma-3-27b-it",
    "dataset": "ECQA (validation)",
    "total_questions": total,
    "correct": correct,
    "accuracy": round(accuracy, 4),
    "extraction_failures": extraction_failures,
    "true_accuracy_excluding_extraction_failures": round(correct / (total - extraction_failures), 4) if total - extraction_failures > 0 else None,
    "per_question": results,
}

output_path = "/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/src/ecqa_eval_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {output_path}")