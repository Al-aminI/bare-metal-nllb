"""
Convert a CTranslate2 int8 model.bin into a single safetensors file.

Usage (from project root):

  python3 ct2_to_safetensors_int8.py \\
      --ct2-dir /tmp/nllb-200-600M-ct2-int8 \\
      --out model_int8_ct2.safetensors

This script:
- Parses CTranslate2's binary model format (model.bin).
- Extracts all variables (including int8 weights) into numpy arrays.
- Writes them into a safetensors file, keeping original variable names.

We will then adapt loader.c to read these int8 weights instead of NF4.
"""

import argparse
import os
import struct
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import save_file


DTYPE_MAP = {
    0: np.float32,  # float32
    1: np.int8,     # int8
    2: np.int16,    # int16
    3: np.int32,    # int32
    4: np.float16,  # float16
    5: np.float16,  # bfloat16 is not needed here; int8 model won't use it
}


def _read_cstring(f) -> str:
    """Read CTranslate2 string format: uint16 len, bytes, trailing NUL."""
    (length_plus_one,) = struct.unpack("H", f.read(2))
    raw = f.read(length_plus_one)
    if not raw or len(raw) != length_plus_one:
        raise RuntimeError("Unexpected EOF while reading string")
    # Drop the trailing NUL.
    return raw[:-1].decode("utf-8")


def load_ct2_model_bin(path: str) -> Tuple[str, Dict[str, np.ndarray]]:
    """
    Parse a CTranslate2 model.bin into a dict: name -> numpy array.

    Returns:
      model_name, variables
    """
    with open(path, "rb") as f:
        (version,) = struct.unpack("I", f.read(4))
        model_name = _read_cstring(f)
        (revision,) = struct.unpack("I", f.read(4))
        (n_vars,) = struct.unpack("I", f.read(4))

        print(f"[ct2-to-st] version={version}, name={model_name}, revision={revision}")
        print(f"[ct2-to-st] variables={n_vars}")

        vars_dict: Dict[str, np.ndarray] = {}

        for i in range(n_vars):
            name = _read_cstring(f)
            (rank,) = struct.unpack("B", f.read(1))
            if rank > 8:
                raise RuntimeError(f"Unreasonable rank {rank} for variable {name}")
            if rank > 0:
                dims = struct.unpack("I" * rank, f.read(4 * rank))
            else:
                dims = ()

            (dtype_id,) = struct.unpack("B", f.read(1))
            (n_bytes,) = struct.unpack("I", f.read(4))

            if dtype_id not in DTYPE_MAP:
                raise RuntimeError(f"Unsupported dtype_id {dtype_id} for {name}")
            np_dtype = DTYPE_MAP[dtype_id]

            raw = f.read(n_bytes)
            if len(raw) != n_bytes:
                raise RuntimeError(f"Unexpected EOF while reading {name}")

            arr = np.frombuffer(raw, dtype=np_dtype)
            if dims:
                arr = arr.reshape(dims)
            else:
                arr = arr.reshape(())

            # Copy so the array does not depend on the mmap'd buffer.
            vars_dict[name] = arr.copy()

            if i < 10:
                print(f"  var[{i}] {name}: dtype={np_dtype}, shape={arr.shape}")

        # Aliases: additional names pointing to existing variables.
        (n_alias,) = struct.unpack("I", f.read(4))
        print(f"[ct2-to-st] aliases={n_alias}")
        for _ in range(n_alias):
            alias = _read_cstring(f)
            target = _read_cstring(f)
            if target not in vars_dict:
                raise RuntimeError(f"Alias target {target} not found for alias {alias}")
            vars_dict[alias] = vars_dict[target]

        return model_name, vars_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2-dir", required=True, help="Path to CTranslate2 model directory")
    ap.add_argument("--out", required=True, help="Output safetensors path")
    args = ap.parse_args()

    model_bin = os.path.join(args.ct2_dir, "model.bin")
    if not os.path.isfile(model_bin):
        raise SystemExit(f"model.bin not found in {args.ct2_dir}")

    print(f"[ct2-to-st] Reading {model_bin} ...")
    model_name, vars_dict = load_ct2_model_bin(model_bin)

    # Optional sanity: show some global stats.
    n_int8 = sum(1 for v in vars_dict.values() if v.dtype == np.int8)
    n_f32 = sum(1 for v in vars_dict.values() if v.dtype == np.float32)
    print(f"[ct2-to-st] vars total={len(vars_dict)}, int8={n_int8}, float32={n_f32}")

    out_path = os.path.abspath(args.out)
    print(f"[ct2-to-st] Writing safetensors to {out_path} ...")
    save_file(vars_dict, out_path)
    print("[ct2-to-st] Done.")


if __name__ == "__main__":
    main()

