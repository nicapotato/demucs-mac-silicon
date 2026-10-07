#!/usr/bin/env bash
# Download the pinned MLX pickle from the immutable GitHub Release named in
# project.conf MODELS_RELEASE_TAG. Exit 2 if that release does not exist yet
# (first bootstrap: convert, then `make release` publishes the tag).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONF="$REPO_ROOT/project.conf"
test -f "$CONF" || { echo "ERROR: missing $CONF" >&2; exit 1; }

TAG="$(grep '^MODELS_RELEASE_TAG=' "$CONF" | cut -d= -f2 | tr -d '[:space:]')"
if [[ -z "$TAG" ]]; then
  echo "ERROR: MODELS_RELEASE_TAG is empty in $CONF" >&2
  exit 1
fi

OUT="${DEMUCS_MLX_CACHE:-$REPO_ROOT/models}"
mkdir -p "$OUT"
DEST="$OUT/htdemucs_6s_mlx.pkl"

origin_to_repo() {
  local url="$1"
  url="${url%.git}"
  url="${url#git@github.com:}"
  url="${url#https://github.com/}"
  printf '%s\n' "$url"
}

GH_REPO="${GITHUB_REPOSITORY:-}"
if [[ -z "$GH_REPO" ]]; then
  origin="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
  GH_REPO="$(origin_to_repo "$origin")"
fi
if [[ -z "$GH_REPO" ]]; then
  echo "ERROR: set GITHUB_REPOSITORY or git remote origin" >&2
  exit 1
fi

if ! gh release view "$TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
  echo "models release ${TAG} is not published on ${GH_REPO} yet" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
gh release download "$TAG" --repo "$GH_REPO" -p 'htdemucs_6s_mlx.pkl' -p 'htdemucs_6s_mlx.pkl.sha256' -D "$tmp"
test -f "$tmp/htdemucs_6s_mlx.pkl"
if [[ -f "$tmp/htdemucs_6s_mlx.pkl.sha256" ]]; then
  (
    cd "$tmp"
    shasum -a 256 -c htdemucs_6s_mlx.pkl.sha256
  )
fi
cp "$tmp/htdemucs_6s_mlx.pkl" "$DEST"
if [[ -f "$tmp/htdemucs_6s_mlx.pkl.sha256" ]]; then
  cp "$tmp/htdemucs_6s_mlx.pkl.sha256" "$OUT/"
fi
echo "fetched $DEST from ${GH_REPO} ${TAG}"
