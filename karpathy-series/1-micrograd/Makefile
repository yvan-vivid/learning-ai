.PHONY: all test typecheck

all: typecheck test

typecheck:
	mypy micrograd tests

test:
	pytest -vv .
