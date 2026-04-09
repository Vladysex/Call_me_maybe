import argparse
from sys import exit
from argparse import ArgumentParser
import json
import numpy as np
import re

from .file_parsing import reading_file, objects_creation
from .llm_handler import StateMachine, States
from llm_sdk import Small_LLM_Model


def parse_arguments() -> argparse.Namespace:
    parser = ArgumentParser()

    parser.add_argument("-fd", "--function-definition-file", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument('-i', '--input', type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument('-o', '--output', type=str,
                        default="data/output/function_calls.json")

    return parser.parse_args()


def build_system_instruction(parsed_functions) -> str:
    functions_context = '\n'.join(
        [f'- {func.name}:'
         f' {getattr(func, "description", "")}' for func in parsed_functions])
    return (
            'You are a strict AI data extractor. Your ONLY job is to extract'
            'parameters for function calling.\n\n'
            'CRITICAL RULES:\n'
            '1. DO NOT execute the function. DO NOT answer the user\'s prompt.'
            'Just extract the exact arguments.\n'
            '2. For regex substitution, the \'replacement\' parameter is ONLY'
            'the new word or character. NEVER return the entire modified'
            'sentence.\n'
            '3. If the user asks to use a symbol name (like \'asterisks\' or'
            '\'spaces\'), use the actual character (like \'*\' or \' \') '
            'in the \'replacement\' parameter.\n\n'
            'EXAMPLES:\n'
            'User: "Replace all numbers in \'test 123\' with \'NUMBERS\'"\n'
            'Correct: "source_string": "test 123", "regex": "[0-9]+",'
            ' "replacement": "NUMBERS"\n\n'
            'User: "Replace all vowels in \'hello\' with \'asterisks\'"\n'
            'Correct: "source_string": "hello", "regex": "[aeiouAEIOU]",'
            '"replacement": "*"\n\n'
            'User: "Substitute \'apple\' with \'orange\' in \''
            'I like apple\'"\n'
            'Correct: "source_string": "I like apple", "regex": '
            '"apple", "replacement": "orange"\n\n'
            f'Available functions:\n{functions_context}\n\n'
            'Extract the parameters now. Do NOT write parameter types.'
        )


def clean_and_parse_json(raw_json_string) -> dict | None:
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
        model, prompt_data, parsed_functions, sys_instruction) -> dict | None:

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
    args = parse_arguments()

    prompts = reading_file(args.input)
    parsed_functions = objects_creation(args.function_definition_file)

    model = Small_LLM_Model()
    sys_instruction = build_system_instruction(parsed_functions)

    all_results = []

    stops = False

    for prompt_data in prompts:
        result = process_single_prompt(
            model, prompt_data, parsed_functions, sys_instruction)

        stops = True
        if stops:
            exit(0)
        if result is not None:
            all_results.append(result)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
