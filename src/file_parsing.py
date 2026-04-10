from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Dict, List, Any
import json


class PropertyDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: str


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., min_length=2)
    description: str = Field(..., min_length=2)
    parameters: Dict[str, PropertyDefinition]
    returns: PropertyDefinition


def reading_file(filename: str) -> Any:
    """Load JSON data from file.

    Returns parsed content, or None when loading/parsing fails.
    """
    try:
        with open(filename, "r", encoding='utf-8') as file:
            dat = json.load(file)
        return dat
    except FileNotFoundError:
        print(f"READ ERROR| File '{filename}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON PARSING ERROR| Broken syntax in file '{filename}")
        print(f"Line -> {e.lineno}, Column -> {e.colno}: {e.msg}")
        return None
    except IsADirectoryError:
        print(f"PATH ERROR| Expected file, but '{filename}' is a directory")
        return None


def objects_creation(filename: str) -> List[FunctionDefinition]:
    """Build validated function definitions from a JSON file."""
    try:
        parsed_data = reading_file(filename)

        if not parsed_data:
            return list()

        list_of_definitions = list()

        for data in parsed_data:
            func_def = FunctionDefinition(**data)
            list_of_definitions.append(func_def)

    except ValidationError as e:
        print("Validation error|", e.errors()[0]['msg'], e.errors()[0]['loc'])
        return list()
    return list_of_definitions
