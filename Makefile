.PHONY: format

format:
	isort lingbot tests
	yapf -i -r lingbot tests
