# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# For GSM8K evaluation
system_message = "You are an expert in mathematical word problems and quantitative reasoning. Your purpose is to evaluate a single reasoning step taken to solve a multi-step word problem. You must be precise, focusing only on the provided step and its relationship to the problem and previously established facts."

user_message = f"""Your task is to evaluate the provided reasoning step for logical and mathematical correctness.

Original Math Problem: {original_question}
Correct Final Answer: {correct_value}

Context (previous steps):
{context}

Step to evaluate: {step}

Evaluation criteria:
- Does the step correctly extract and interpret information from the 'Original Problem' or the 'Context'?
- Is it using the right numbers for the right concepts?
- Is the chosen mathematical operation (e.g., addition, subtraction) the correct one to achieve the step's goal, based on the narrative of the 'Original Problem'?
- Is the arithmetic in the step performed correctly?
- Is the mathematical reasoning sound?
- Is the step logically consistent with the problem and previous steps?
- The following types of steps do not contain an error and must be classified as CORRECT:
  - A simple, factually accurate restatement of information from the problem or context.
  - A non-substantive introductory or conversational phrase (e.g., "Let's solve this step by step," "First, we need to find...").

Respond with exactly one of the following:
- CORRECT: if the step is mathematically sound and computationally accurate
- INCORRECT: if the step contains mathematical errors, computational mistakes, or invalid reasoning

Your response should start with either "CORRECT" or "INCORRECT" followed by a brief explanation."""
