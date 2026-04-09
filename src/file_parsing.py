from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Any
import json


class PropertyDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str = Field(..., min_length=2)
    description: str = Field(..., min_length=2)
    parameters: Dict[str, PropertyDefinition]
    returns: PropertyDefinition


def reading_file(filename: str) -> Any:
    try:
        with open(filename, "r") as file:
            dat = json.load(file)
        return dat
    except FileNotFoundError as e:
        print("FileNotFoundError|", e)
        return None
    except json.JSONDecodeError as e:
        print("json.JSONDecodeError|", e)
        return None
    except IsADirectoryError as e:
        print("IsADirectoryError|", e)
        return None


def objects_creation(filename: str) -> List[FunctionDefinition]:
    try:
        parsed_data = reading_file(filename)

        if not parsed_data:
            return list()

        list_of_definitions = list()

        for data in parsed_data:
            func_def = FunctionDefinition(**data)
            list_of_definitions.append(func_def)
        return list_of_definitions

    except ValidationError as e:
        print("Validation error|", e)
        return list()
