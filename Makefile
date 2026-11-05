.PHONY: test demo lint

test:
	python -m pytest tests/ -q

demo:
	python demo.py

lint:
	python -m compileall flp demo.py
