.PHONY: install test lint fmt index kg-up

install:
	uv sync

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests

index:
	@echo "make index: not implemented until Phase 1 (see docs/02_ROADMAP.md)"
	@exit 1

kg-up:
	@echo "make kg-up: not implemented until Phase 2 (see docs/02_ROADMAP.md)"
	@exit 1
