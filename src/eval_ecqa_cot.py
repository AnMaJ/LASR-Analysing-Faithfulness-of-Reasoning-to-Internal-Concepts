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
ecqa_data = pd.read_parquet(
    "/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/"
    "Investigating_concept_neurons/"
    "LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/"
    "datasets/ecqa/data/validation-00000-of-00001.parquet"
)
print(f"Dataset size: {len(ecqa_data)} questions")

# ── 3. Build CoT prompt ─────────────────────────────────────────────────────
OPTION_LABELS = ["A", "B", "C", "D", "E"]

def build_cot_prompt(row):
    """Build a chain-of-thought prompt for the given ECQA row."""
    options_text = "\n".join(
        f"{OPTION_LABELS[i].lower()}) {row[f'q_op{i+1}']}" for i in range(5)
    )
    return (
        f"Please think step by step before giving your final answer. "
        f"Consider what information is provided and what assumptions might be involved. "
        f"After your reasoning, clearly state your final answer with ONLY the letter "
        f"(A, B, C, D, or E) of the correct option.\n\n"
        f"Question: {row['q_text']}\n\n"
        f"Options:\n{options_text}"
    )

# ── 4. Extract predicted letter from CoT output ─────────────────────────────
def extract_answer_letter_cot(text: str) -> str | None:
    """
    Extract the final answer letter from a CoT response.
    Tries multiple patterns, prioritising markers near the end of the text.
    """
    text_upper = text.upper().strip()

    # 1. Look for explicit "final answer" markers (most reliable)
    #    Patterns: "Final Answer: A", "The answer is B", "Answer: C", "my answer is D"
    final_patterns = [
        r'FINAL\s+ANSWER\s*[:\-\s]*\(?([A-E])\)?',
        r'THE\s+ANSWER\s+IS\s*[:\-\s]*\(?([A-E])\)?',
        r'(?:MY\s+)?ANSWER\s*[:\-\s]+\(?([A-E])\)?',
        r'CORRECT\s+(?:OPTION|ANSWER)\s*[:\-\s]*\(?([A-E])\)?',
        r'I\s+(?:WOULD\s+)?CHOOSE\s*[:\-\s]*\(?([A-E])\)?',
    ]
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, text_upper))
        if matches:
            return matches[-1].group(1)  # take the LAST match (closest to end)

    # 2. Look for a standalone letter near the very end of the response
    #    Check the last 50 characters for a clear letter
    tail = text_upper[-50:]
    tail_match = re.search(r'\b([A-E])\b(?:\s*[\.\):]?\s*$)', tail)
    if tail_match:
        return tail_match.group(1)

    # 3. Look for bold/emphasised letter patterns: **A**, *A*
    bold_match = re.search(r'\*\*([A-E])\*\*', text_upper)
    if bold_match:
        return bold_match.group(1)

    # 4. Fallback: last standalone A-E letter in the entire response
    all_matches = re.findall(r'\b([A-E])\b', text_upper)
    if all_matches:
        return all_matches[-1]

    return None

# ── 5. Get the correct option's letter ───────────────────────────────────────
def get_correct_letter(row):
    """Match q_ans text to one of q_op1–q_op5 and return the letter."""
    answer = row["q_ans"].strip().lower()
    for i in range(5):
        if row[f"q_op{i+1}"].strip().lower() == answer:
            return OPTION_LABELS[i]
    return None

# ── 6. Run CoT evaluation ───────────────────────────────────────────────────
results = []
correct = 0
total = 0

for idx, row in tqdm(ecqa_data.iterrows(), total=len(ecqa_data), desc="Evaluating (CoT)"):
    prompt = build_cot_prompt(row)
    correct_letter = get_correct_letter(row)

    # Tokenize using chat template
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    # Generate with more tokens for CoT reasoning
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,      # allow space for reasoning + answer
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    predicted_letter = extract_answer_letter_cot(response)
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
print(f"Final CoT Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")

# Count extraction failures
no_answer = sum(1 for r in results if r["predicted_letter"] is None)
print(f"Extraction failures (no letter found): {no_answer}/{total}")

output = {
    "model": "google/gemma-3-27b-it",
    "dataset": "ECQA (validation)",
    "prompt_type": "chain-of-thought",
    "total_questions": total,
    "correct": correct,
    "accuracy": round(accuracy, 4),
    "extraction_failures": no_answer,
    "true_accuracy_excluding_extraction_failures": round(correct / (total - no_answer), 4) if total - no_answer > 0 else None,
    "per_question": results,
}

output_path = "/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/src/ecqa_cot_eval_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {output_path}")