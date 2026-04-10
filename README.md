*This project has been created as part of the 42 curriculum by <b>vlprysia</b>.*

# Call-Me-Maybe

## Description
Call-Me-Maybe is a constrained function-calling pipeline for local LLM inference.

Goal:
- Convert a natural language user prompt into a strict function-call JSON object.
- Keep generation valid and structured even when the model would normally drift or produce malformed output.

The project combines:
- JSON function schema parsing with validation (Pydantic models).
- A token-level state machine that constrains decoding.
- A cleaning/recovery step for common malformed JSON edge cases.
- Test coverage for parsing, extraction behavior, and cleanup logic.

Expected output shape per prompt:
- prompt
- name
- parameters

## Features
- Deterministic token forcing for fixed JSON segments.
- Prefix-constrained function name generation.
- Parameter key forcing based on selected function schema.
- Type-aware value stopping rules (string vs number-like values).
- Recovery cleaning for malformed braces/comma patterns.
- CLI arguments for input, schema, and output paths.

## Instructions
### Requirements
- Python >= 3.10
- uv (recommended by this project Makefile)

### Installation
1. Install dependencies:
   - make install

Or manually:
- uv sync

### Run
Default run:
- make run

Equivalent command:
- uv run python -m src

Custom files:
- uv run python -m src -fd data/input/functions_definition.json -i data/input/function_calling_tests.json -o data/output/function_calls.json

### Debug
- make debug

### Lint
- make lint
- make lint-strict

### Test
- make test

## Algorithm Explanation (Constrained Decoding)
The core decoding logic is implemented as a finite state machine that controls which tokens are legal at each generation step.

High-level flow:
1. Build a system instruction from prompt template + available functions list.
2. For each user prompt, initialize a state machine with function definitions.
3. Start token generation from the instruction context.
4. At each step, ask the state machine for allowed tokens.
5. Decode using one of three policies:
   - Single allowed token: force it.
   - Multiple allowed tokens: mask logits to only allowed tokens, then argmax.
   - Free generation (values): allow all tokens and argmax.
6. Update state with the generated token/text.
7. Stop once all required keys/values are produced.
8. Clean and parse generated JSON.

State details:
- EXPECTING_PROMPT_KEY: force literal prefix for prompt field.
- EXPECTING_FUNCTION_NAME: allow only tokens that keep a valid function-name prefix.
- EXPECTING_PARAMETERS_KEY: force literal parameters key opening.
- FORCING_PARAMETER_KEY: force each parameter key according to schema order.
- EXPECTING_PARAMETERS_VALUE: free generation for value content until stop trigger.
- DONE: finalize object.

Why this works:
- Structural tokens are never guessed.
- Function name cannot leave the known set.
- Parameter keys are schema-driven.
- Only value regions are left open, reducing error surface.

## Design Decisions
- Finite State Machine over regex-only post-processing:
  - Prevents many invalid structures before they happen.
- Pydantic schema models:
  - Enforces strict function-definition input (extra fields forbidden).
- Argmax decoding with token masking:
  - Keeps behavior deterministic and easier to debug.
- Separate cleanup stage (clean_and_parse_json):
  - Pragmatic fallback for residual model formatting mistakes.
- JSON-safe prompt insertion:
  - Prompt text is encoded safely before forcing output prefix.

## Performance Analysis
### Accuracy
- Function-name selection is strongly improved by prefix-constrained decoding.
- Parameter-key correctness is high because keys are forced from schema.
- Value extraction quality still depends on model understanding of prompt semantics.

### Speed
- Constraint checks are lightweight compared to model forward pass.
- Extra state transitions add small overhead but generally improve success rate, reducing retries.

### Reliability
- Structural reliability is high due to forced tokens.
- Cleanup layer improves robustness against malformed tail output.
- Remaining fragility is mostly in ambiguous user prompts and free-form value generation.

## Challenges Faced
1. Malformed JSON from the model around braces/commas.
2. Keeping generation flexible for values but strict for structure.
3. Correct stopping behavior for string vs non-string parameter types.
4. Balancing strictness and practical recovery logic.

How they were handled:
- Introduced constrained state transitions for structure.
- Added targeted cleanup regex + tail normalization for common corruption patterns.
- Used type-aware stop triggers in value state.
- Added focused tests for malformed JSON recovery.

## Problem Example: Bad Curly-Bracket Behavior
A common failure mode is premature brace closure while parameters are still being generated.

Expected:
{
  "parameters": {
    "a": 2,
    "b": 3
  }
}

Malformed model output example:
{
  "parameters": {
    "a": 2}},
    "b": 3
}

Why it happens:
- The model emits an extra closing brace before finishing sibling parameters.

Mitigation in this project:
- Constrained generation reduces when/where closing braces can appear.
- clean_and_parse_json applies normalization and attempts safe reconstruction before JSON parsing.

## Testing Strategy
The project uses pytest-based automated checks with three complementary layers:

1. Parsing and schema validation tests
- Validate valid definitions, invalid schema handling, missing file behavior, and empty list handling.

2. Constrained function-calling behavior tests
- Parameterized prompts verify the predicted function name for multiple intents.
- Ensures generated result includes a parameters dictionary.

3. Cleanup robustness tests
- Validate normal JSON parsing.
- Validate recovery from malformed brace/comma patterns.
- Validate trailing comma/bracket cleanup behavior.

How to run:
- make test
- or: uv run pytest tests/ -v

## Example Usage
Input prompt list:
- data/input/function_calling_tests.json

Function definitions:
- data/input/functions_definition.json

Run extraction:
- uv run python -m src -fd data/input/functions_definition.json -i data/input/function_calling_tests.json -o data/output/function_calls.json

Example output item:
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}

## Project Structure
- src/file_parsing.py: input reading + schema object creation.
- src/llm_handler.py: constrained decoding state machine.
- src/test_main.py: CLI, generation loop, cleanup, output write.
- data/prompt.txt: system instruction template.
- tests/: parsing, cleanup, and behavior tests.

## Resources
Classic references:
- JSON format: https://www.rfc-editor.org/rfc/rfc8259
- Python argparse: https://docs.python.org/3/library/argparse.html
- Python json: https://docs.python.org/3/library/json.html
- Pydantic docs: https://docs.pydantic.dev/
- pytest docs: https://docs.pytest.org/
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers/index

How AI was used:
- AI assistance was used for:
  - Drafting and refining documentation wording.
  - Brainstorming edge cases for malformed JSON outputs.
  - Suggesting additional test scenarios for constrained decoding behavior.
- Final code and documentation decisions were reviewed and validated manually through tests and project-specific constraints.
