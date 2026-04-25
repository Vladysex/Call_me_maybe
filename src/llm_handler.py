import json
from typing import List, Any
from enum import Enum, auto

from llm_sdk import Small_LLM_Model
from src.file_parsing import FunctionDefinition


class States(Enum):
    EXPECTING_PROMPT_KEY = auto()
    EXPECTING_FUNCTION_NAME = auto()
    EXPECTING_PARAMETERS_KEY = auto()
    FORCING_PARAMETER_KEY = auto()
    EXPECTING_PARAMETERS_VALUE = auto()
    DONE = auto()


class StateMachine:
    def __init__(self, model: Small_LLM_Model, original_user_prompt: str,
                 list_of_function: List[FunctionDefinition]) -> None:
        """Initialize FSM state and pre-tokenized forced JSON fragments."""

        self.list_of_function = list_of_function
        self.original_user_prompt = original_user_prompt
        list_of_function_names = [func.name for func in list_of_function]

        self.current_state = States.EXPECTING_PROMPT_KEY
        self.current_step = 0

        safe_prompt = json.dumps(original_user_prompt)
        target_string = f'{{"prompt":{safe_prompt},\n "name": "'
        parameter_string = '",\n "parameters":{'

        self.prompt_key_ids = model.encode(target_string)[0].tolist()
        self.list_of_ids = [
            model.encode(name.strip())[0].tolist()
            for name in list_of_function_names
            ]
        self.list_of_parameter_ids = model.encode(parameter_string)[0].tolist()

        self.current_name_tokens: List[int] = []
        self.parameter_step = 0
        self.chosen_function: Any = None

        self.param_names: List[str] = []
        self.current_param_index = 0
        self.current_param_type: Any = None
        self.current_param_key_ids: List[int] = []
        self.param_key_step = 0

    def get_allowed_tokens(self) -> List[int] | str:
        """Return allowed next token ids for
        the current decoding state."""
        if self.current_state == States.EXPECTING_PROMPT_KEY:
            return [self.prompt_key_ids[self.current_step]]

        elif self.current_state == States.EXPECTING_FUNCTION_NAME:
            allowed_next_tokens = set()
            prefix_len = len(self.current_name_tokens)
            for func_id in self.list_of_ids:
                if len(func_id) > prefix_len and func_id[
                        :prefix_len] == self.current_name_tokens:
                    allowed_next_tokens.add(func_id[prefix_len])
            return list(allowed_next_tokens)

        elif self.current_state == States.EXPECTING_PARAMETERS_KEY:
            return [self.list_of_parameter_ids[self.parameter_step]]

        elif self.current_state == States.FORCING_PARAMETER_KEY:
            return [self.current_param_key_ids[self.param_key_step]]

        elif self.current_state == States.EXPECTING_PARAMETERS_VALUE:
            return 'ALL'

        return []

    def advance_state(self, generated_token_id: int,
                      generated_text: str, model: Small_LLM_Model) -> None:
        """Update state after one token
        and dispatch the right handler."""

        if self.current_state == States.EXPECTING_PROMPT_KEY:
            self._handle_prompt_key()
        elif self.current_state == States.EXPECTING_FUNCTION_NAME:
            self._handle_function_name(generated_token_id)
        elif self.current_state == States.EXPECTING_PARAMETERS_KEY:
            self._handle_parameters_key(model)
        elif self.current_state == States.FORCING_PARAMETER_KEY:
            self._handle_forcing_parameter_key()
        elif self.current_state == States.EXPECTING_PARAMETERS_VALUE:
            self._handle_parameters_value(generated_text, model)

    def _handle_prompt_key(self) -> None:
        """Advance through the forced
        JSON prefix before function name decoding."""
        self.current_step += 1
        if self.current_step >= len(self.prompt_key_ids):
            self.current_state = States.EXPECTING_FUNCTION_NAME

    def _handle_function_name(self, generated_token_id: int) -> None:
        """Accumulate function-name tokens and lock on full match."""
        self.current_name_tokens.append(generated_token_id)

        if self.current_name_tokens in self.list_of_ids:
            func_index = self.list_of_ids.index(self.current_name_tokens)
            self.chosen_function = self.list_of_function[func_index]
            self.current_state = States.EXPECTING_PARAMETERS_KEY

    def _handle_parameters_key(self, model: Small_LLM_Model) -> None:
        """Finalize parameters opening
        and prepare the first key."""

        self.parameter_step += 1
        if self.parameter_step >= len(self.list_of_parameter_ids):
            self.param_names = list(self.chosen_function.parameters.keys())
            self.current_param_index = 0

            if self.param_names:
                self._prepare_next_parameter_key(model, prefix_str="")
            else:
                self.current_state = States.DONE

    def _handle_forcing_parameter_key(self) -> None:
        """Consume forced parameter-key tokens,
        then switch to value generation."""
        self.param_key_step += 1
        if self.param_key_step >= len(self.current_param_key_ids):

            self.current_state = States.EXPECTING_PARAMETERS_VALUE

    def _handle_parameters_value(
            self, generated_text: str, model: Small_LLM_Model
    ) -> None:
        """Detect value boundary
        and move to next key or finish."""
        stop_triggered = False

        if self.current_param_type == 'string':
            if '"' in generated_text:
                stop_triggered = True
        else:
            if ',' in generated_text or '}' in generated_text:
                stop_triggered = True

        if stop_triggered:
            self.current_param_index += 1

            if self.current_param_index < len(self.param_names):
                prefix = ' ' if ',' in generated_text else ', '
                self._prepare_next_parameter_key(model, prefix_str=prefix)
            else:
                self.current_state = States.DONE

    def _prepare_next_parameter_key(
            self, model: Small_LLM_Model, prefix_str: str
    ) -> None:
        """Create forced tokens for
        the next schema parameter key and type context."""
        next_param_name = self.param_names[self.current_param_index]
        current_param = self.chosen_function.parameters[next_param_name]
        self.current_param_type = current_param.type
        if self.current_param_type == 'string':
            target_key_string = f'{prefix_str}"{next_param_name}": "'
        else:
            target_key_string = f'{prefix_str}"{next_param_name}": '

        self.current_param_key_ids = model.encode(
            target_key_string)[0].tolist()
        self.param_key_step = 0
        self.current_state = States.FORCING_PARAMETER_KEY
