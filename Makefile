.PHONY: install serve test
install:
	python -m pip install -e '.[dev]'
serve:
	uvicorn carsel.api:app --reload --host 127.0.0.1 --port 8000
test:
	pytest -q
