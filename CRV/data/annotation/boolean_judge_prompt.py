# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# For boolean expressions
system_message = "You are an expert in logical reasoning and boolean algebra. You evaluate the correctness of reasoning steps in boolean expression evaluation with high precision."

user_message = f"""Evaluate this reasoning step for logical correctness:

Original Boolean Expression: {original_expression}
Correct Truth Value: {correct_value}

Context (previous steps):
{context}

Step to evaluate: {step}

Evaluation criteria:
- Is the boolean operation applied correctly?
- Does the step follow proper order of operations?
- Are the truth values computed accurately?
- Is the reasoning logically sound?

Respond with exactly one of the following:
- CORRECT: if the step is logically sound and mathematically accurate
- INCORRECT: if the step contains logical errors, mathematical mistakes, or invalid reasoning

Your response should start with either "CORRECT" or "INCORRECT" followed by a brief explanation."""
