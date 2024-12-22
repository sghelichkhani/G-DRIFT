.PHONY: lint test longtest longtest_output convert_demos

lint:
	@echo "Linting module code"
	@python3 -m flake8 gdrift 
	@echo "Linting examples and tests"
	@python3 -m flake8 examples tests
