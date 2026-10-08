#!/usr/bin/env python3
"""Export htdemucs_6s MLX pickle weights to safetensors + JSON for the C worker."""
from __future__ import annotations

import argparse
import json
import sys
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
        help="Model name (default: htdemucs_6s)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=str(ROOT / "models"),
        help="Output directory",
    )
    args = parser.parse_args(argv)

    from demucs_mlx.c_export import (
        flatten_params,
        model_config_dict,
        save_safetensors,
        unwrap_htdemucs,
        write_keys_manifest,
    )
    from demucs_mlx.model_converter import get_mlx_model

    model = unwrap_htdemucs(get_mlx_model(args.name))
    if hasattr(model, "eval"):
        model.eval()
    tensors = flatten_params(model.parameters())
    if not tensors:
        raise SystemExit("no parameters found on model")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name
    st_path = out_dir / f"{stem}.safetensors"
    cfg_path = out_dir / f"{stem}.json"
    keys_path = out_dir / f"{stem}.keys.txt"

    save_safetensors(st_path, tensors)
    cfg = model_config_dict(model)
    cfg["weights"] = st_path.name
    cfg["num_tensors"] = len(tensors)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    write_keys_manifest(keys_path, tensors)

    total = sum(int(t.size) for t in tensors.values())
    print(f"wrote {st_path} ({len(tensors)} tensors, {total} scalars)")
    print(f"wrote {cfg_path}")
    print(f"wrote {keys_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
