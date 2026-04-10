import pytest
from src.file_parsing import objects_creation
from pydantic import ValidationError


def test_objects_creation_success(tmp_path):

    d = tmp_path / "input"
    d.mkdir()
    f = d / "functions.json"
    f.write_text("""
    [
        {
            "name": "fn_add",
            "description": "Adds two numbers",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number"}
        }
    ]
    """)

    functions = objects_creation(str(f))

    assert len(functions) == 1
    assert functions[0].name == "fn_add"
    assert functions[0].parameters["a"].type == "number"


def test_objects_creation_invalid_json(tmp_path):
    f = tmp_path / "invalid_schema.json"
    f.write_text("""
    [
        {
            "name": "fn_fail",
            "parameters": {},
            "returns": {"type": "string"}
        }
    ]
    """)

    functions = objects_creation(str(f))

    assert functions == []


def test_objects_creation_file_not_found():
    functions = objects_creation("ghost_file.json")

    assert functions == []


def test_objects_creation_empty_list(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text("[]")

    functions = objects_creation(str(f))
    assert functions == []