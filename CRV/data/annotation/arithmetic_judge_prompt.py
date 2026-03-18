# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# For arithmetic expressions
system_message = "You are an expert in mathematical reasoning and arithmetic operations. You evaluate the correctness of reasoning steps in arithmetic expression evaluation with high precision."

user_message = f"""Evaluate this reasoning step for mathematical correctness:

Original Arithmetic Expression: {original_expression}
Correct Value: {correct_value}

Context (previous steps):
{context}

Step to evaluate: {step}

Evaluation criteria:
- Are the arithmetic operations applied correctly?
- Does the step follow proper order of operations (PEMDAS/BODMAS)?
- Are the numerical computations accurate?
- Is the mathematical reasoning sound?

Respond with exactly one of the following:
- CORRECT: if the step is mathematically sound and computationally accurate
- INCORRECT: if the step contains mathematical errors, computational mistakes, or invalid reasoning

Your response should start with either "CORRECT" or "INCORRECT" followed by a brief explanation."""
