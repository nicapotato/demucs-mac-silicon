#!/usr/bin/env python3
"""Export htdemucs_6s from the upstream demucs-mlx cache for the C worker.

Writes ``models/htdemucs_6s.{safetensors,json,keys.txt}`` and
``models/export_manifest.json`` (upstream version, hashes, mlx / mlx-c pairing).

``--check`` exports into a temp dir and fails if the tensor names/shapes, config or
weight bytes differ from the committed manifest, without touching ``models/``. Run it
after every upstream bump (``make verify`` does).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--name",
        default="htdemucs_6s",
        choices=["htdemucs_6s"],
        help="Model name; the C worker only supports htdemucs_6s",
    )
    parser.add_argument("-o", "--out", default=str(ROOT / "models"), help="Output directory")
    parser.add_argument("--check", action="store_true", help="Compare against committed files")
    parser.add_argument(
        "--allow-version-change",
        action="store_true",
        help="With --check: tolerate a different demucs-mlx version (upgrade flow)",
    )
    args = parser.parse_args(argv)

    # Keep upstream's cache apart from the C export (same filename, different keys).
    os.environ.setdefault("DEMUCS_MLX_CACHE_DIR", str(ROOT / "models" / "mlx-cache"))

    from worker_py import contract
    from worker_py.c_export import export_model

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            res = export_model(args.name, Path(tmp))
            problems = contract.check_against_committed(
                res, allow_version_change=args.allow_version_change
            )
        for p in problems:
            print(f"DRIFT: {p}", file=sys.stderr)
        print("export matches committed manifest" if not problems else "export DRIFTED")
        return 1 if problems else 0

    res = export_model(args.name, Path(args.out))
    print(f"wrote {res['safetensors']} ({res['num_tensors']} tensors)")
    print(f"wrote {res['config']}")
    print(f"wrote {res['keys']}")
    if Path(args.out).resolve() == (ROOT / "models").resolve():
        manifest = contract.build_manifest(
            cache_file=res["cache_file"],
            export_file=res["safetensors"],
            num_tensors=res["num_tensors"],
        )
        contract.write_manifest(manifest)
        print(f"wrote {contract.MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
