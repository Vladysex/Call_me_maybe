.PHONY: install run debug clean lint lint-strict test
DOP_LIBRARIES = pydantic numpy pytest

install:
	uv sync
	uv add $(DOP_LIBRARIES)


run:
	  	uv run python -m src


debug:
		uv run python -m pdb src

lint:
	uv run flake8 src
	uv run mypy -m src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	  flake8 --exclude=llm_sdk src/
	  uv run mypy -m src --strict

test:
		uv run pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
