PYTHON=python3
TEST_DIR=tests
REQUIREMENTS=requirements.txt

.PHONY: all clean test install run

all:
	$(PYTHON) -m main
	find . -name '__pycache__' -exec rm -rf {} +

test:
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p '*_test.py'

install:
	$(PYTHON) -m pip install -r $(REQUIREMENTS)
