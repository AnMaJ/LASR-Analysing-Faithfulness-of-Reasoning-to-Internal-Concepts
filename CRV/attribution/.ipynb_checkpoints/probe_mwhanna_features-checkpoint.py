"""
Probe the mwhanna features/ binary format.

Steps:
  1. Download index.json.gz (~27 MB) once and cache it.
  2. Show the structure of the index for a sample feature.
  3. Fetch the first 512 bytes of layer_0.bin via HTTP range request.
  4. Print a hex dump so we can reverse-engineer the record format.

Run:
    python CRV/attribution/probe_mwhanna_features.py
"""

from __future__ import annotations

import gzip
import json
import struct
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_BASE = "https://huggingface.co/mwhanna/gemma-scope-2-1b-it/resolve/main"
FEATURES_PATH = "transcoder_all/width_262k_l0_small_affine/features"
INDEX_URL = f"{HF_BASE}/{FEATURES_PATH}/index.json.gz"
BIN_URL_TEMPLATE = f"{HF_BASE}/{FEATURES_PATH}/layer_{{layer}}.bin"

CACHE_DIR = Path.home() / ".cache" / "mwhanna_features"
INDEX_CACHE = CACHE_DIR / "index.json.gz"

PROBE_LAYER = 0
PROBE_FEATURE = 0   # first feature in the index
PROBE_BYTES = 512   # bytes to fetch for format inspection


# ---------------------------------------------------------------------------
# 1. Download / cache the index
# ---------------------------------------------------------------------------

def get_index() -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_CACHE.exists():
        print(f"Downloading index from {INDEX_URL} …", flush=True)
        resp = requests.get(INDEX_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with INDEX_CACHE.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"Saved to {INDEX_CACHE} ({INDEX_CACHE.stat().st_size:,} bytes)")
    else:
        print(f"Using cached index: {INDEX_CACHE}")

    with gzip.open(INDEX_CACHE, "rb") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Inspect index structure
# ---------------------------------------------------------------------------

def inspect_index(index: dict) -> None:
    print("\n--- Index top-level keys ---")
    for k in list(index.keys())[:20]:
        print(f"  {k!r}: {type(index[k]).__name__}")

    print(f"\nTotal top-level entries: {len(index)}")

    # Try to find something that looks like layer → feature → offset
    sample_key = next(iter(index))
    sample_val = index[sample_key]
    print(f"\nSample entry  key={sample_key!r}  value type={type(sample_val).__name__}")
    if isinstance(sample_val, dict):
        sub_key = next(iter(sample_val))
        print(f"  sub-key={sub_key!r}  sub-value={sample_val[sub_key]!r}")
    elif isinstance(sample_val, (list, tuple)):
        print(f"  first element: {sample_val[0]!r}")
    else:
        print(f"  value: {sample_val!r}")


# ---------------------------------------------------------------------------
# 3. Fetch a range from a .bin file
# ---------------------------------------------------------------------------

def range_fetch(url: str, start: int, length: int) -> bytes:
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {resp.status_code} for range request: {url}")
    return resp.content


def hex_dump(data: bytes, width: int = 16) -> None:
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:04x}  {hex_part:<{width * 3}}  {ascii_part}")


def try_parse_record(data: bytes) -> None:
    """Heuristic attempts to decode the first record."""
    print("\n--- Heuristic parse attempts ---")

    # 4-byte little-endian int at offset 0
    if len(data) >= 4:
        val = struct.unpack_from("<I", data, 0)[0]
        print(f"  u32 LE @ 0: {val}")

    # 8-byte little-endian int at offset 0
    if len(data) >= 8:
        val = struct.unpack_from("<Q", data, 0)[0]
        print(f"  u64 LE @ 0: {val}")

    # Maybe a JSON header?
    try:
        text = data.decode("utf-8", errors="replace")
        if text.startswith("{") or text.startswith("["):
            print(f"  Looks like JSON: {text[:200]}")
    except Exception:
        pass

    # Maybe MessagePack? (first byte 0x8X = fixmap, 0x9X = fixarray)
    if data:
        print(f"  First byte: 0x{data[0]:02x}  "
              f"({'fixmap' if 0x80 <= data[0] <= 0x8f else 'fixarray' if 0x90 <= data[0] <= 0x9f else '?'})")


# ---------------------------------------------------------------------------
# 4. Find a real feature offset from the index
# ---------------------------------------------------------------------------

def find_offset(index: dict, layer: int, feature_id: int) -> int | None:
    """Try common index shapes to find the byte offset for (layer, feature_id)."""
    # Shape A: index[str(layer)][str(feature_id)] = offset
    if str(layer) in index:
        sub = index[str(layer)]
        if isinstance(sub, dict) and str(feature_id) in sub:
            return sub[str(feature_id)]

    # Shape B: index[str(feature_id)] = {str(layer): offset}
    if str(feature_id) in index:
        sub = index[str(feature_id)]
        if isinstance(sub, dict) and str(layer) in sub:
            return sub[str(layer)]

    # Shape C: index[str(layer)] = list of offsets indexed by feature_id
    if str(layer) in index:
        sub = index[str(layer)]
        if isinstance(sub, list) and feature_id < len(sub):
            return sub[feature_id]

    # Shape D: flat list
    if isinstance(index, list) and feature_id < len(index):
        return index[feature_id]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def decode_record(data: bytes) -> dict:
    """Parse one record: 4-byte LE size header + gzip-compressed JSON body."""
    if len(data) < 4:
        raise ValueError("Record too short for header")
    compressed_size = struct.unpack_from("<I", data, 0)[0]
    compressed = data[4: 4 + compressed_size]
    return json.loads(gzip.decompress(compressed))


def fetch_feature(layer: int, feature_id: int, index: dict) -> dict | None:
    """Fetch and decode one feature record from the mwhanna features store.

    Index shape:
        index[str(layer)] = {
            "filename": "layer_X.bin",
            "offsets": list[int | None]  # indexed by feature_id; None = dead feature
        }
    Record shape: [4 bytes uint32 LE compressed_size][gzip JSON bytes]
    """
    layer_entry = index.get(str(layer))
    if layer_entry is None:
        return None
    offsets = layer_entry.get("offsets", [])
    if feature_id >= len(offsets):
        return None
    offset = offsets[feature_id]
    if offset is None:
        return None

    bin_url = BIN_URL_TEMPLATE.format(layer=layer)
    header = range_fetch(bin_url, offset, 4)
    compressed_size = struct.unpack_from("<I", header, 0)[0]
    body = range_fetch(bin_url, offset + 4, compressed_size)
    return json.loads(gzip.decompress(body))


def main() -> None:
    index = get_index()
    print(f"\nversion: {index.get('version')}")
    print(f"format:  {index.get('format')}")

    layer0_entry = index.get("0", {})
    offsets = layer0_entry.get("offsets", [])
    print(f"\nLayer 0: {len(offsets)} feature slots")

    # Find first non-None feature
    first_fid = next((i for i, v in enumerate(offsets) if v is not None), None)
    if first_fid is None:
        print("No active features in layer 0.")
        sys.exit(1)
    print(f"First active feature: {first_fid} → byte offset {offsets[first_fid]}")

    # Also show a few more
    active = [(i, v) for i, v in enumerate(offsets) if v is not None]
    print(f"Total active features in layer 0: {len(active)}")
    print(f"Sample (fid, offset): {active[:5]}")

    # Decode the first active feature
    print(f"\n--- Decoding layer=0 feature={first_fid} ---")
    record = fetch_feature(0, first_fid, index)
    if record is None:
        print("Fetch failed.")
        sys.exit(1)

    print(f"Record keys: {list(record.keys())}")
    text = json.dumps(record, indent=2)
    print(text[:3000])
    if len(text) > 3000:
        print("  … (truncated)")


if __name__ == "__main__":
    main()
