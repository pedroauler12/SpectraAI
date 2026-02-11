.PHONY: setup-dev test coverage

setup-dev:
	python -m pip install -U pip
	python -m pip install -r requirements-dev.txt

test:
	pytest -q testes

coverage:
	pytest -q --cov=src --cov-report=term-missing testes
