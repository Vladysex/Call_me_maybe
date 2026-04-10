import pytest
from src.main import process_single_prompt, build_system_instruction
from src.file_parsing import objects_creation
from llm_sdk import Small_LLM_Model


@pytest.fixture(scope="module")
def llm_model():
    return Small_LLM_Model()


@pytest.fixture(scope="module")
def parsed_funcs():
    return objects_creation("data/input/functions_definition.json")


@pytest.fixture(scope="module")
def sys_instr(parsed_funcs):
    return build_system_instruction(parsed_funcs)


@pytest.mark.parametrize("user_prompt, expected_function_name", [
    ("What is the sum of 265 and 345?", "fn_add_numbers"),
    ("Greet shrek", "fn_greet"),
    ("Calculate the square root of 144", "fn_get_square_root"),
    ("Replace all numbers in 'test 123' with NUMBERS", "fn_substitute_string_with_regex")
])
def test_llm_predicts_correct_function(user_prompt, expected_function_name, llm_model, parsed_funcs, sys_instr):

    prompt_data = {"prompt": user_prompt}

    result = process_single_prompt(llm_model, prompt_data, parsed_funcs, sys_instr)

    assert result is not None

    assert result["name"] == expected_function_name

    assert isinstance(result.get("parameters"), dict)