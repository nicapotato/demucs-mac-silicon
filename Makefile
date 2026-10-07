# MLX worker — tests, freeze, CI dispatch.

CI_WORKFLOW := .github/workflows/ci.yml

REPO ?= $(shell git remote get-url origin 2>/dev/null | sed -E 's|git@github\.com:||; s|https://github\.com/||; s|\.git$$||')
GH_R := $(if $(REPO),-R "$(REPO)",)
REF ?= $(shell git branch --show-current 2>/dev/null)
PYTHON := $(CURDIR)/.venv/bin/python

.PHONY: test lint worker clean ci ci-watch

lint:
	uv run ruff check demucs_mlx tests
	uv run pyright

test: lint
	uv run python tests/test_metal_kernels.py
	uv run python tests/test_apply_model_chunk_seed.py
	uv run python tests/test_model_converter_optional_mlx_weights.py
	uv run python tests/test_apply_model_overlap_add.py
	uv run demucs-mlx --list-models

worker:
	DEMUCS_MLX_PYTHON="$(PYTHON)" bash scripts/freeze_worker.sh

clean:
	rm -rf dist .ruff_cache .pytest_cache

ci:
	@test -n "$(REPO)" || (echo "ERROR: could not resolve origin repo; set REPO=owner/name" >&2; exit 1)
	gh $(GH_R) workflow run "$(CI_WORKFLOW)" \
		$(if $(REF),-r "$(REF)",) \
		$(if $(VERSION),-f version="$(VERSION)",)

ci-watch: ci
	@sleep 2
	@RID=$$(gh $(GH_R) run list --workflow="$(CI_WORKFLOW)" -L 1 --json databaseId -q '.[0].databaseId'); \
		test -n "$$RID"; \
		gh $(GH_R) run watch "$$RID"
