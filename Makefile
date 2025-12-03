.PHONY: format lint typecheck test build clean

format:
	black agentforge tests

lint:
	flake8 agentforge tests

typecheck:
	mypy agentforge

test:
	pytest

build:
	python -m build

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete 