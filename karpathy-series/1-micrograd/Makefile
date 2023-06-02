.PHONY: all test typecheck

all: typecheck test

typecheck:
	poetry run mypy micrograd tests

test:
	poetry run pytest -vv .
