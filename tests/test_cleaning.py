import pytest
from src.main import clean_and_parse_json


def test_clean_and_parse_json_valid():
    raw_input = '{"name": "fn_greet", "parameters": {"name": "shrek"}}'
    result = clean_and_parse_json(raw_input)
    assert result["name"] == "fn_greet"
    assert result["parameters"]["name"] == "shrek"


def test_clean_and_parse_json_with_garbage():

    raw_input = '{"name": "fn_add", "parameters": {"a": 5}} , "b": 10'
    result = clean_and_parse_json(raw_input)

    assert result is not None
    assert result["parameters"]["a"] == 5
    assert result["parameters"]["b"] == 10


def test_clean_and_parse_json_trailing_comma():
    raw_input = '{"name": "fn_test", "parameters": {"a": 1},}\n'
    result = clean_and_parse_json(raw_input)
    assert result is not None