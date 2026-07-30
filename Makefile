.PHONY: install demo test check

install:
	python -m pip install -e .

demo:
	deepsearch-demo

test:
	python -m unittest discover -s tests -v

check:
	python -m compileall -q src scripts tests
	python -m unittest discover -s tests -v
