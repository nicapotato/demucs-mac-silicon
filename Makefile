# demucs-mac-silicon: worker_py (wrapper over pinned demucs-mlx) + worker-c (mlx-c).
#
#   make verify         everything that must pass before bumping upstream / releasing
#   make upgrade-check  after editing the demucs-mlx pin: lock, sync, verify
#
# See docs/upgrading.md.

CI_WORKFLOW := .github/workflows/ci.yml
RELEASE_WORKFLOW := .github/workflows/release.yml

REPO ?= $(shell git remote get-url origin 2>/dev/null | sed -E 's|git@github\.com:||; s|https://github\.com/||; s|\.git$$||')
GH_R := $(if $(REPO),-R "$(REPO)",)
REF ?= $(shell git branch --show-current 2>/dev/null)
PYTHON := $(CURDIR)/.venv/bin/python
PUBLISH_GH_RELEASE ?= true
JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Upstream weights cache (safetensors, digest-checked by demucs-mlx). Kept apart from the
# C worker export in models/ because both use <model>.safetensors with different keys.
MLX_CACHE := $(CURDIR)/models/mlx-cache
export DEMUCS_MLX_CACHE_DIR := $(MLX_CACHE)

.PHONY: lint test test-fast sync native-deps verify upgrade-check models-fetch models-export models-check models-check-upgrade \
	goldens bench worker worker-py worker-pyinstaller worker-py-verify-build worker-c \
	worker-c-test clean ci ci-watch release release-watch

sync:
	uv sync --extra dev --extra freeze

# worker-c --mp3 needs a static libmp3lame (Homebrew lame >= 4 also needs mpg123).
native-deps:
	@command -v brew >/dev/null || (echo "ERROR: Homebrew required for worker-c (libmp3lame)" >&2; exit 1)
	brew list lame >/dev/null 2>&1 || brew install lame
	brew list mpg123 >/dev/null 2>&1 || brew install mpg123

lint:
	uv run ruff check worker_py tests scripts
	uv run pyright

# Everything that needs no weights, binaries or network.
test-fast: lint
	uv run pytest tests -q -m "not model"

# Full suite. Model / C-worker / frozen-worker tests skip when their artifact is missing;
# `make verify` builds them all first so nothing skips.
test: lint
	uv run pytest tests -q -rs

models-fetch:
	@if [ -f "$(MLX_CACHE)/htdemucs_6s.safetensors" ] && [ -f "$(MLX_CACHE)/htdemucs_6s_config.json" ]; then \
		echo "weights present in $(MLX_CACHE)"; \
	else \
		mkdir -p "$(MLX_CACHE)"; \
		DEMUCS_MLX_NO_DOWNLOAD=0 uv run python -m demucs_mlx.hub download htdemucs_6s --output-dir "$(MLX_CACHE)"; \
	fi

# Re-export the C worker's weights/config/keys + models/export_manifest.json (commit the diff).
models-export: models-fetch
	uv run python scripts/export_safetensors.py -n htdemucs_6s -o models

# Fail if the export no longer matches the committed manifest (upstream drift).
models-check: models-fetch
	uv run python scripts/export_safetensors.py --check

# Upgrade flow only: like models-check but tolerates the demucs-mlx version differing.
models-check-upgrade: models-fetch
	uv run python scripts/export_safetensors.py --check --allow-version-change

goldens: models-fetch
	uv run python scripts/dump_goldens.py -n htdemucs_6s

bench:
	uv run python tests/bench_compare.py --durations 10 $(if $(C_WORKER),--c-worker "$(C_WORKER)",) $(if $(FROZEN),--frozen "$(FROZEN)",)

worker-c:
	cmake -S worker-c -B worker-c/build -DCMAKE_BUILD_TYPE=Release
	cmake --build worker-c/build -j $(JOBS)
	test -x worker-c/build/demucs_mlx_worker

worker-c-test: worker-c
	./worker-c/build/test_kernels
	./worker-c/build/test_stft
	./worker-c/build/test_ola
	@if [ -f models/htdemucs_6s.safetensors ] && [ -f goldens/htdemucs_6s/goldens.safetensors ]; then \
		./worker-c/build/test_compare models/htdemucs_6s.safetensors models/htdemucs_6s.json goldens/htdemucs_6s/goldens.safetensors goldens/htdemucs_6s/mix.wav; \
	else \
		echo "skip test_compare (export models + goldens first)"; \
	fi

# Stage the C worker for demucs-ui-app.
worker: worker-c
	mkdir -p dist/worker
	cp worker-c/build/demucs_mlx_worker dist/worker/demucs_mlx_worker
	cp worker-c/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib dist/worker/mlx.metallib
	test -x dist/worker/demucs_mlx_worker

# Frozen worker_py (PyInstaller onedir) -> dist/worker/demucs_mlx_worker/ (release layout).
worker-pyinstaller worker-py:
	DEMUCS_MLX_PYTHON="$(PYTHON)" bash scripts/freeze_worker.sh

# Same, into dist/worker-py so it does not clobber the staged C worker. Used by verify/CI.
worker-py-verify-build:
	DEMUCS_MLX_PYTHON="$(PYTHON)" DEMUCS_MLX_WORKER_DIST="$(CURDIR)/dist/worker-py" bash scripts/freeze_worker.sh

# The gate. Order: cheap static checks first, drift check before anything expensive.
# goldens are regenerated (not committed) so worker-c test_compare actually runs on CI.
verify: lint models-check models-export goldens worker-c worker-py-verify-build
	$(MAKE) worker-c-test
	uv run pytest tests -q -rs
	@echo "verify OK: upstream $$(uv run python -c 'import demucs_mlx;print(demucs_mlx.__version__)')"

# After changing the demucs-mlx pin in pyproject.toml: relock, make sure the C worker's
# contract (tensors/config/bytes) survived, re-record the manifest, then run the full gate.
# A contract break stops here (fix worker-c / re-export deliberately, do not paper over it).
upgrade-check:
	uv lock
	uv sync --extra dev --extra freeze
	$(MAKE) models-fetch models-check-upgrade
	$(MAKE) models-export
	$(MAKE) verify
	@git --no-pager diff --stat -- pyproject.toml uv.lock models/ || true

clean:
	rm -rf dist .ruff_cache .pytest_cache worker-c/build

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

release:
	@test -n "$(REPO)" || (echo "ERROR: could not resolve origin repo; set REPO=owner/name" >&2; exit 1)
	gh $(GH_R) workflow run "$(RELEASE_WORKFLOW)" \
		$(if $(REF),-r "$(REF)",) \
		-f publish_gh_release="$(PUBLISH_GH_RELEASE)" \
		$(if $(VERSION),-f version="$(VERSION)",)

release-watch: release
	@sleep 2
	@RID=$$(gh $(GH_R) run list --workflow="$(RELEASE_WORKFLOW)" -L 1 --json databaseId -q '.[0].databaseId'); \
		test -n "$$RID"; \
		gh $(GH_R) run watch "$$RID"
