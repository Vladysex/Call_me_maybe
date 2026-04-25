# Call-Me-Maybe

*This project has been created as part of the 42 curriculum by <b>vlprysia</b>.*

---

## 📌 Description
Call-Me-Maybe is a constrained function-calling pipeline for local LLM inference.

### Goal
- Convert a natural language user prompt into a strict function-call JSON object.
- Keep generation valid and structured even when the model would normally drift or produce malformed output.

---

## 🚀 Features
- Deterministic token forcing for fixed JSON segments
- Prefix-constrained function name generation
- Parameter key forcing based on selected function schema
- Type-aware value stopping rules (string vs number-like values)
- Recovery cleaning for malformed JSON edge cases
- CLI arguments for input, schema, and output paths

---

## 📦 Expected Output Shape

```json
{
  "prompt": "...",
  "name": "...",
  "parameters": {
    "...": "..."
  }
}
```

---

## ⚙️ Installation

### Requirements
- Python >= 3.10
- uv (recommended)

### Install dependencies

```bash
make install
```

or manually:

```bash
uv sync
```

---

## ▶️ Run

Default:

```bash
make run
```

Equivalent:

```bash
uv run python -m src
```

Custom files:

```bash
uv run python -m src \
  -fd data/input/functions_definition.json \
  -i data/input/function_calling_tests.json \
  -o data/output/function_calls.json
```

---

## 🐛 Debug

```bash
make debug
```

---

## 🧹 Lint

```bash
make lint
make lint-strict
```

---

## 🧪 Test

```bash
make test
```

or

```bash
uv run pytest tests/ -v
```

---

## 🧠 Algorithm Explanation (Constrained Decoding)

The core decoding logic is implemented as a **finite state machine** that controls which tokens are allowed at each generation step.

### High-level flow

1. Build system instruction from prompt template + available functions
2. Initialize state machine with function definitions
3. Start token generation
4. At each step → get allowed tokens from FSM
5. Decode using:
   - Single token → force
   - Multiple tokens → mask + argmax
   - Free generation → argmax
6. Update state
7. Stop when all required fields are generated
8. Clean and parse JSON

---

## 🔄 State Machine

- `EXPECTING_PROMPT_KEY` → force `"prompt"`
- `EXPECTING_FUNCTION_NAME` → prefix-constrained selection
- `EXPECTING_PARAMETERS_KEY` → force `"parameters"`
- `FORCING_PARAMETER_KEY` → enforce schema keys
- `EXPECTING_PARAMETERS_VALUE` → free generation
- `DONE` → finalize

---

## 💡 Why This Works

- Structural tokens are never guessed
- Function names are restricted to valid prefixes
- Parameter keys are schema-driven
- Only values are generated freely

---

## 🧪 Problem Example: Bad Curly-Bracket Behavior

### Expected

```json
{
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

### Malformed Output

```json
{
  "parameters": {
    "a": 2}},
    "b": 3
}
```

### Cause
- Model closes braces too early

### Fix
- Constrained decoding limits structure errors
- Cleanup step repairs malformed JSON

---

## 🧪 Testing Strategy

### 1. Parsing & Schema Validation
- Valid/invalid schema handling
- Missing file behavior
- Empty definitions

### 2. Function Selection
- Prompt → correct function mapping
- Ensures parameters field exists

### 3. Cleanup Robustness
- Valid JSON parsing
- Broken JSON recovery
- Trailing comma fixes

---

## 📥 Example Usage

### Run extraction

```bash
uv run python -m src \
  -fd data/input/functions_definition.json \
  -i data/input/function_calling_tests.json \
  -o data/output/function_calls.json
```

### Example Output

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

---

## 📁 Project Structure

- `src/file_parsing.py` → schema parsing
- `src/llm_handler.py` → FSM decoding
- `src/test_main.py` → CLI & pipeline
- `data/prompt.txt` → system prompt
- `tests/` → unit tests

---

## 📚 Resources

- JSON: https://www.rfc-editor.org/rfc/rfc8259
- argparse: https://docs.python.org/3/library/argparse.html
- json: https://docs.python.org/3/library/json.html
- Pydantic: https://docs.pydantic.dev/
- pytest: https://docs.pytest.org/
- Transformers: https://huggingface.co/docs/transformers

---

## 🤖 AI Usage

AI was used for:
- Documentation drafting
- Edge case brainstorming
- Test ideas

All outputs were manually validated and tested.