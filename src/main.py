import argparse
from argparse import ArgumentParser
import json
import numpy as np
import re
import os
import sys
from typing import List, Any

from .file_parsing import reading_file, objects_creation, FunctionDefinition
from .llm_handler import StateMachine, States
from llm_sdk import Small_LLM_Model


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for function
    definitions, input prompts, and output path."""
    parser = ArgumentParser()

    parser.add_argument("-fd", "--function-definition-file", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument('-i', '--input', type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument('-o', '--output', type=str,
                        default="data/output/function_calls.json")

    return parser.parse_args()


def build_system_instruction(
        parsed_functions: List[FunctionDefinition]) -> str:
    """Build the system prompt from template and function descriptions.
    text and available function metadata."""
    functions_context = '\n'.join(
        [f'- {func.name}:'
         f' {getattr(func, "description", "")}' for func in parsed_functions])
    try:
        with open('data/prompt.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print('Critical error: Prompt file not found.')
        sys.exit(1)

    final_prompt = (f'{content}\n'
                    f'Available functions:\n{functions_context}\n\n'
                    f'Extract the parameters now. '
                    f'Do NOT write parameter types.')
    return final_prompt


def clean_and_parse_json(raw_json_string: str) -> dict[str, Any] | None | Any:
    """Normalize malformed JSON fragments and parse the result.
    patterns and parse into a Python object."""
    cleaned_string = re.sub(r'\}\}?\s*,\s*"', ', "', raw_json_string)
    cleaned_string = cleaned_string.rstrip(' ,}\n')
    cleaned_string += '}}'
    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        print("\nParsing ERROR:", e)
        print("Bad string:\n", cleaned_string)
        return None


def process_single_prompt(
        model: Small_LLM_Model,
        prompt_data: dict[str, str],
        parsed_functions: List[FunctionDefinition],
        sys_instruction: str
) -> dict[str, Any] | None:
    """Generate one constrained function-call object for a prompt.
    JSON object for a single user prompt."""

    state_machine = StateMachine(
        model=model,
        original_user_prompt=prompt_data['prompt'],
        list_of_function=parsed_functions
    )

    input_ids = model.encode(sys_instruction)[0].tolist()
    instruction_length = len(input_ids)

    while state_machine.current_state != States.DONE:
        allowed_tokens = state_machine.get_allowed_tokens()

        if allowed_tokens == 'ALL':
            logits = model.get_logits_from_input_ids(input_ids)
            generated_token_id = int(np.argmax(logits))

        elif isinstance(allowed_tokens, list) and len(allowed_tokens) == 1:
            generated_token_id = allowed_tokens[0]

        elif isinstance(allowed_tokens, list):
            logits = model.get_logits_from_input_ids(input_ids)
            logits_array = np.array(logits)
            inf_logits = np.full_like(logits_array, fill_value=-np.inf)

            for token_id in allowed_tokens:
                inf_logits[token_id] = logits_array[token_id]

            generated_token_id = int(np.argmax(inf_logits))

        input_ids.append(generated_token_id)
        generated_text = model.decode([generated_token_id])

        print(generated_text, end="", flush=True)

        state_machine.advance_state(generated_token_id, generated_text, model)

    final_json_string = model.decode(input_ids[instruction_length:])
    return clean_and_parse_json(final_json_string)


def main() -> None:
    """Run the pipeline and write extracted calls to output JSON.
    and save extracted calls to output JSON."""
    args = parse_arguments()

    prompts = reading_file(args.input)

    if not prompts:
        print('Critical error. There are no input prompts to process.')
        sys.exit(1)

    if not isinstance(prompts, list):
        print("Critical error: Json file has to be a list.")
        sys.exit(1)

    parsed_functions = objects_creation(args.function_definition_file)
    if not isinstance(parsed_functions, list):
        print("Critical error: Json file has to be a list.")
        sys.exit(1)

    if not parsed_functions:
        print("Critical error: parsed_functions is empty or falsy.")
        sys.exit(1)

    model = Small_LLM_Model()
    sys_instruction = build_system_instruction(parsed_functions)

    all_results = []

    for prompt_data in prompts:
        result = process_single_prompt(
            model, prompt_data, parsed_functions, sys_instruction)

        if result is not None:
            all_results.append(result)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
