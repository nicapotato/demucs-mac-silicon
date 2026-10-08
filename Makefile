# MLX worker — tests, freeze, CI / release dispatch.

CI_WORKFLOW := .github/workflows/ci.yml
RELEASE_WORKFLOW := .github/workflows/release.yml

REPO ?= $(shell git remote get-url origin 2>/dev/null | sed -E 's|git@github\.com:||; s|https://github\.com/||; s|\.git$$||')
GH_R := $(if $(REPO),-R "$(REPO)",)
REF ?= $(shell git branch --show-current 2>/dev/null)
PYTHON := $(CURDIR)/.venv/bin/python
PUBLISH_GH_RELEASE ?= true

.PHONY: test lint worker models-fetch models-convert models-export goldens bench worker-c worker-c-test clean ci ci-watch release release-watch

lint:
	uv run ruff check demucs_mlx tests
	uv run pyright

test: lint
	uv run python tests/test_metal_kernels.py
	uv run python tests/test_apply_model_chunk_seed.py
	uv run python tests/test_model_converter_optional_mlx_weights.py
	uv run python tests/test_apply_model_overlap_add.py
	uv run python tests/test_c_export.py
	uv run demucs-mlx --list-models

worker: worker-c
	mkdir -p dist/worker
	cp worker-c/build/demucs_mlx_worker dist/worker/demucs_mlx_worker
	cp worker-c/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib dist/worker/mlx.metallib
	test -x dist/worker/demucs_mlx_worker

worker-pyinstaller:
	DEMUCS_MLX_PYTHON="$(PYTHON)" bash scripts/freeze_worker.sh

models-fetch:
	bash scripts/fetch_mlx_weights.sh

models-convert:
	@test -x "$(PYTHON)" || (echo "ERROR: missing $(PYTHON) — uv sync --extra convert" >&2; exit 1)
	mkdir -p models
	DEMUCS_MLX_CACHE="$(CURDIR)/models" "$(PYTHON)" -c "from demucs_mlx.model_converter import get_mlx_model; get_mlx_model('htdemucs_6s'); print('ok')"
	test -f models/htdemucs_6s_mlx.pkl

models-export:
	mkdir -p models
	DEMUCS_MLX_CACHE="$(CURDIR)/models" uv run python scripts/export_safetensors.py -n htdemucs_6s -o models

goldens: models-export
	DEMUCS_MLX_CACHE="$(CURDIR)/models" uv run python scripts/dump_goldens.py -n htdemucs_6s

bench:
	DEMUCS_MLX_CACHE="$(CURDIR)/models" uv run python tests/bench_compare.py --durations 10 $(if $(C_WORKER),--c-worker "$(C_WORKER)",)

worker-c:
	cmake -S worker-c -B worker-c/build -DCMAKE_BUILD_TYPE=Release
	cmake --build worker-c/build -j
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
