# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
import re
from typing import List, Dict, Optional, Union
import os

def parse_cot_steps(cot_string: str) -> List[Dict[str, Union[str, int, None]]]:
    """
    Parse chain of thought string into step-level components.

    Args:
        cot_string: The chain of thought reasoning string

    Returns:
        List of dictionaries, each containing:
        - step_number: int (0 for initial text, then numbered steps)
        - content: str (the full step content)
    """

    steps = []
    lines = cot_string.strip().split('\n')

    # Patterns for numbered steps
    numbered_step_pattern = r'^(\d+)\.\s*(.+)$'
    markdown_step_pattern = r'^##\s*Step\s*(\d+):\s*(.+)$'

    # First, collect all initial text before numbered steps
    initial_text_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # If we hit a numbered step (either format), stop collecting initial text
        if (re.match(numbered_step_pattern, line) or
            re.match(markdown_step_pattern, line, re.IGNORECASE)):
            break

        initial_text_lines.append(line)
        i += 1

    # Add initial text as step 0 if it exists
    if initial_text_lines:
        steps.append({
            'step_number': 0,
            'content': '\n'.join(initial_text_lines)
        })

    # Process numbered steps
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check for numbered steps (both formats)
        numbered_match = re.match(numbered_step_pattern, line)
        markdown_match = re.match(markdown_step_pattern, line, re.IGNORECASE)

        step_match = numbered_match or markdown_match
        if step_match:
            step_num = int(step_match.group(1))

            # Collect all content for this step
            step_content_lines = [line]
            j = i + 1

            # Look ahead for content that belongs to this step
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue

                # If we hit another numbered step (either format), stop
                if (re.match(numbered_step_pattern, next_line) or
                    re.match(markdown_step_pattern, next_line, re.IGNORECASE)):
                    break

                # If we hit a conclusion line (starts with "Therefore" or contains "final result"), stop
                if (next_line.lower().startswith('therefore') or
                    'final result' in next_line.lower() or
                    'final answer' in next_line.lower()):
                    break

                # Include this line as part of the current step
                step_content_lines.append(next_line)
                j += 1

            steps.append({
                'step_number': step_num,
                'content': '\n'.join(step_content_lines)
            })

            i = j
        else:
            # Handle any remaining text (like conclusion lines)
            remaining_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line:
                    remaining_lines.append(next_line)
                j += 1

            if remaining_lines:
                # Find the next available step number
                next_step_num = max([s['step_number'] for s in steps if s['step_number'] is not None], default=0) + 1
                steps.append({
                    'step_number': next_step_num,
                    'content': '\n'.join(remaining_lines)
                })
            break

    return steps

def extract_boolean_result(steps: List[Dict]) -> Optional[bool]:
    """
    Extract the final boolean result from parsed steps.

    Args:
        steps: List of parsed step dictionaries

    Returns:
        Boolean result or None if not found
    """
    # Look through steps in reverse order to find the final result
    for step in reversed(steps):
        content = step.get('content', '').lower()

        # Look for explicit final result statements
        if 'final result' in content or 'therefore' in content or 'final answer' in content:
            if 'true' in content:
                return True
            elif 'false' in content:
                return False

    # If no explicit final answer found, look for boolean results in last step
    step = steps[-1]
    content = step.get('content', '').lower()

    # Look for evaluation results like "=true", "= true", "0", "1", or other eligible answers
    if any(keyword in content for keyword in ['=true', '= true', 'true', '1']):
        return True
    elif any(keyword in content for keyword in ['=false', '= false', 'false', '0']):
        return False

    return None

def _safe_eval(expr: str) -> Optional[Union[int, float]]:
    """
    Helper function to safely evaluate a string expression using asteval.
    It cleans the expression and handles potential evaluation errors.
    """
    if not isinstance(expr, str):
        return None

    cleaned_expr = expr.strip()

    # asteval can't handle single hyphens or empty strings
    if cleaned_expr == "-" or not cleaned_expr:
        return None

    try:
        # Use a fresh, safe evaluator for each expression
        result = eval(cleaned_expr)
        if isinstance(result, (int, float)):
            return result
    except (SyntaxError, NameError, TypeError, Exception):
        # Catch a wide range of evaluation errors
        pass
    return None

def extract_arithmetic_result(steps: List[Dict]) -> Optional[Union[int, float]]:
    """
    Extracts the final arithmetic result from parsed steps by finding the
    last evaluatable expression in the final step.

    Args:
        steps: List of parsed step dictionaries.

    Returns:
        The numeric result (int or float), or None if not found.
    """
    # This pattern is designed to find the last expression on a line.
    # It captures numbers or anything inside parentheses at the end of a string.
    final_expression_pattern = re.compile(
        r".*?([-]?\s*(?:\d+(?:\.\d+)?|\(.*\)))\.?\s*$"
    )

    for step in reversed(steps):
        content = step.get('content', '').strip()
        if not content:
            continue

        # 1. High Priority: Check for a LaTeX \boxed{} expression first.
        boxed_match = re.search(r'\\boxed\{(.+?)\}', content, re.DOTALL)
        if boxed_match:
            result = _safe_eval(boxed_match.group(1))
            if result is not None:
                return result

        # 2. Heuristic: Check the last non-empty line for a final expression.
        last_line = next((line for line in reversed(content.split('\n')) if line.strip()), None)

        if last_line:
            match = final_expression_pattern.match(last_line.strip())
            if match:
                # The regex captures the last potential expression on the line
                potential_expr = match.group(1)
                result = _safe_eval(potential_expr)
                if result is not None:
                    return result

    return None


def display_parsed_steps(steps: List[Dict]) -> None:
    """Display parsed steps in a readable format."""
    for i, step in enumerate(steps):
        print(f"Step {i+1}:")
        print(f"  Step Number: {step['step_number']}")
        print(f"  Content: {step['content']}")
        print()


def scan_directories_and_load_data_boolean(base_path=None, filename="boolean_expressions_data.json", bsz=10):
    all_conversations = []
    conversation_id = 0
    for i in range(10):
        for j in range(bsz):
            current_path = f"{base_path}.{i}/{j}/answered_dialogs.pt"
            try:
                current_data = torch.load(current_path)

                # Process each conversation
                for conversation in current_data:
                    conversation_messages = []

                    # Extract and save the different parts
                    for msg in conversation:
                        role = msg['role']

                        message_info = {
                            "role": role,
                            "content": msg['content']
                        }

                        # If this is an assistant message, parse the chain of thought
                        if role == 'assistant':
                            parsed_steps = parse_cot_steps(msg['content'])
                            predicted_result = extract_boolean_result(parsed_steps)
                            message_info["predicted_truth_value"] = predicted_result
                            message_info["step_level"] = parsed_steps

                        # Add additional fields if they exist
                        if 'expression_id' in msg:
                            message_info['expression_id'] = conversation_id

                        if 'truth_value' in msg:
                            message_info['truth_value'] = msg['truth_value']

                        conversation_messages.append(message_info)

                    all_conversations.append(conversation_messages)
                    conversation_id += 1

            except FileNotFoundError:
                print(f"File not found: {current_path}")
            except Exception as e:
                print(f"Error loading {current_path}: {str(e)}")

    # Ensure the directory for the filename exists
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(all_conversations, f, indent=2)

    print(f"Data saved to {filename} with {len(all_conversations)} conversations")
    return all_conversations

def scan_directories_and_load_data_arithmetic(base_path=None, filename="arithmetic_expressions_data.json", bsz=10):
    all_conversations = []
    conversation_id = 0
    for j in range(bsz):
        current_path = f"{base_path}/{j}/answered_dialogs.pt"
        try:
            current_data = torch.load(current_path)

            # Process each conversation
            for conversation in current_data:
                conversation_messages = []

                # Extract and save the different parts
                for msg in conversation:
                    role = msg['role']

                    message_info = {
                        "role": role,
                        "content": msg['content']
                    }

                    # If this is an assistant message, parse the chain of thought
                    if role == 'assistant':
                        parsed_steps = parse_cot_steps(msg['content'])
                        predicted_result = extract_arithmetic_result(parsed_steps)
                        message_info["predicted_value"] = predicted_result
                        message_info["step_level"] = parsed_steps

                    # Add additional fields if they exist
                    if 'expression_id' in msg:
                        message_info['expression_id'] = conversation_id

                    if 'value' in msg:
                        message_info['value'] = msg['value']

                    conversation_messages.append(message_info)

                all_conversations.append(conversation_messages)
                conversation_id += 1

        except FileNotFoundError:
            print(f"File not found: {current_path}")
        except Exception as e:
            print(f"Error loading {current_path}: {str(e)}")

    # Ensure the directory for the filename exists
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(all_conversations, f, indent=2)

    print(f"Data saved to {filename} with {len(all_conversations)} conversations")
    return all_conversations

if __name__ == "__main__":
    model_name = "Llama-3.1-8B-Instruct"

    expressions_type = "arithmetic_expressions" # "boolean_expressions" or "arithmetic_expressions"

    expr_type = "arith" # "arith" or "bool"

    for nt, nd, bsz in zip(["nt3", "nt5", "nt7", "nt10"], ["nd10000", "nd10000", "nd10000", "nd10000"], [400, 400, 400, 400]):
        path = f"path/to/llama_dumps/meta-llama/{model_name}/{expressions_type}/{expr_type}.{nd}.{nt}"
        scan_directories_and_load_data_arithmetic(base_path=path,
                                    filename=f"data/{model_name}/{expressions_type}/{expr_type}.{nd}.{nt}.json",
                                    bsz=bsz)
