# Dataset Generation and Annotation

The full processed datasets used in the paper (including step-level annotations) are hosted on [Hugging Face](https://huggingface.co/datasets/facebook/crv).


Alternatively, if you wish to generate the data from scratch, follow the following steps.

## 1. Expression Generation

We utilize three datasets: Synthetic Arithmetic, Synthetic Boolean, and GSM8K.

### Synthetic Datasets
We provide scripts in the `generation/` folder to create controlled expressions.

**Arithmetic Expressions:**
```bash
python generation/generate_arithmetic_expressions.py
```
- **Configuration:** You can modify the script to change:
    - `num_dialogs`: Number of expressions to generate
    - `nonterminal_nodes`: Number of operators to use (e.g. `+, -, *, and, or`)

**Boolean Expressions:**
```bash
python generation/generate_boolean_expressions.py
```
- *Note:* For low operator counts (e.g., n=3), the script uses exhaustive search to generate all possible unique expressions. For higher counts, it switches to sampling.

**Output:** Files are saved to `data/` with naming conventions reflecting their complexity (e.g., `arith.nd10000.nt10` for arithmetic expressions with 10000 expressions and 10 operators).

### GSM8K

The third dataset is the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset. We use the test split for our experiments.

## 2. Generating Chain-of-Thoughts (Inference)
Once the generation of expressions are done, next step is to evaluate the LLMs on these expressions and generate Chain-of-Thoughts steps. To run eval on arithmetic expressions, use the following command

```bash
cd processing/
python eval_arith_expressions.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_dir /path/to/CRV/data/bool.nd1000.nt10 \
    --answer_prefix llama_dumps/ \
    --job_id $SLURM_ARRAY_TASK_ID \
    --random_seed 2806 \
    --temperature 0.1 \
    --no_dialogs
```
*Note that the script is optimized to use `SLURM Job Arrays` to run things in parallel, if you don't run things in parallel, you will need to set the `job_id` manually, and modify the logic with batching.*

### Segmentation
After inference, the generated CoTs will be saved under `llama_dumps/`. Use the processing script to parse the raw text output into individual CoT reasoning steps:

```bash
python process_cot_steps.py
```

## 3. Annotation
We propose two methods to generate ground-truth labels for reasoning steps.

### Method 1: LLM-as-a-Judge
We verify steps using a stronger model (e.g., Llama 3.3 70B Instruct).
- **Prompts:** The exact prompts used for Boolean, Arithmetic, and GSM8K are provided in the `annotation/` folder.
- You may use these templates with any capable LLM API.

### Method 2: Programmatic State Verification
For synthetic tasks, we use a heuristic approach described in **Appendix A.2** of the paper.
- **Mechanism:** We prompt the model to output a simplified state (e.g., "*Now the original expression becomes...*") after every step.
- **Verification:** We programmatically evaluate this simplified expression. If its value diverges from the original ground truth, the step is marked as incorrect.

*Note: In the paper, we generated final labels by taking the intersection of Method 1 and Method 2 to ensure high-fidelity annotations.*
