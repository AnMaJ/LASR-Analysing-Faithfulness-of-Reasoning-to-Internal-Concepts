"""
Random-access reader for the mwhanna Gemma Scope 2 feature examples store.

The store lives at:
  https://huggingface.co/mwhanna/gemma-scope-2-1b-it/resolve/main/
      transcoder_all/width_262k_l0_small_affine/features/

Format (discovered by probing):
  index.json.gz  — maps layer → list of byte offsets (indexed by feature_id)
  layer_X.bin    — concatenated records:
                     [4 bytes uint32 LE: compressed_size][gzip JSON]

Each JSON record has:
  transcoder_id       str
  index               int   (feature id)
  examples_quantiles  list of {quantile_name, examples}
    examples:
      tokens           list[str]   — token strings in the context window
      tokens_acts_list list[float] — activation per token
      train_token_ind  int         — index of the peak-activation token
  top_logits     list[str]  — vocabulary items the feature promotes
  bottom_logits  list[str]  — vocabulary items the feature suppresses
  act_max        float
  activation_frequency float

Usage
-----
    store = MwhannaFeatureStore()           # loads + caches index (~27 MB)
    info  = store.fetch(layer=13, feature_id=186358)
    print(info["top_logits"])
    for ex in info["examples"]:
        print(ex["context"])                # text snippet around peak activation
        print(ex["max_activation"])
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_BASE = (
    "https://huggingface.co/mwhanna/gemma-scope-2-1b-it/resolve/main"
    "/transcoder_all/width_262k_l0_small_affine/features"
)
INDEX_URL = f"{HF_BASE}/index.json.gz"
BIN_URL_TEMPLATE = f"{HF_BASE}/layer_{{layer}}.bin"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mwhanna_features"
CONTEXT_RADIUS = 8   # tokens before/after peak to include in snippet


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MwhannaFeatureStore:
    """Lazily-loaded, cached reader for the mwhanna feature examples store."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict | None = None   # loaded on first use

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        index_path = self._cache_dir / "index.json.gz"
        if not index_path.exists():
            print(f"[MwhannaFeatureStore] Downloading index (~27 MB) …", flush=True)
            resp = requests.get(INDEX_URL, stream=True, timeout=120)
            resp.raise_for_status()
            with index_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            print(f"[MwhannaFeatureStore] Index saved to {index_path}")
        with gzip.open(index_path, "rb") as f:
            return json.load(f)

    @property
    def index(self) -> dict:
        if self._index is None:
            self._index = self._load_index()
        return self._index

    # ------------------------------------------------------------------
    # Byte-range fetch
    # ------------------------------------------------------------------

    @staticmethod
    def _range_fetch(url: str, start: int, length: int) -> bytes:
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status_code} fetching range from {url}")
        return resp.content

    # ------------------------------------------------------------------
    # Raw record fetch
    # ------------------------------------------------------------------

    def _fetch_raw(self, layer: int, feature_id: int) -> dict | None:
        layer_entry = self.index.get(str(layer))
        if layer_entry is None:
            return None
        offsets: list = layer_entry.get("offsets", [])
        if feature_id >= len(offsets):
            return None
        offset = offsets[feature_id]
        if offset is None:
            return None

        url = BIN_URL_TEMPLATE.format(layer=layer)
        header = self._range_fetch(url, offset, 4)
        compressed_size = struct.unpack_from("<I", header, 0)[0]
        body = self._range_fetch(url, offset + 4, compressed_size)
        return json.loads(gzip.decompress(body))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        layer: int,
        feature_id: int,
        n_examples: int = 5,
    ) -> dict[str, Any] | None:
        """Return a clean summary dict for one feature, or None if not found.

        Returned dict keys:
            layer          int
            feature_id     int
            act_max        float
            activation_frequency  float
            top_logits     list[str]
            bottom_logits  list[str]
            examples       list[{context, max_activation, peak_token}]
        """
        raw = self._fetch_raw(layer, feature_id)
        if raw is None:
            return None

        examples = []
        for quantile_block in raw.get("examples_quantiles", []):
            if quantile_block.get("quantile_name") != "Top":
                continue
            for ex in quantile_block.get("examples", [])[:n_examples]:
                tokens: list[str] = ex.get("tokens", [])
                acts: list[float] = ex.get("tokens_acts_list", [])
                peak: int = ex.get("train_token_ind", 0)

                lo = max(0, peak - CONTEXT_RADIUS)
                hi = min(len(tokens), peak + CONTEXT_RADIUS + 1)
                context_tokens = tokens[lo:hi]
                context = "".join(context_tokens).replace("\u23ce", "\n")

                max_act = max(acts) if acts else 0.0
                peak_token = tokens[peak] if peak < len(tokens) else ""

                examples.append({
                    "context": context,
                    "max_activation": max_act,
                    "peak_token": peak_token,
                })
            break  # only process "Top" quantile

        return {
            "layer": layer,
            "feature_id": feature_id,
            "act_max": raw.get("act_max", 0.0),
            "activation_frequency": raw.get("activation_frequency", 0.0),
            "top_logits": raw.get("top_logits", [])[:10],
            "bottom_logits": raw.get("bottom_logits", [])[:5],
            "examples": examples,
        }

    def format_for_print(self, info: dict, rank: int | None = None) -> str:
        """Format a fetch() result as a human-readable string."""
        prefix = f"[{rank}] " if rank is not None else ""
        lines = [
            f"{prefix}L{info['layer']}/F{info['feature_id']}  "
            f"act_max={info['act_max']:.2f}  "
            f"freq={info['activation_frequency']:.4f}",
        ]
        if info["top_logits"]:
            lines.append(f"  Top logits:    {info['top_logits']}")
        if info["bottom_logits"]:
            lines.append(f"  Bottom logits: {info['bottom_logits']}")
        for i, ex in enumerate(info["examples"], 1):
            ctx = ex["context"].replace("\n", "↵")
            lines.append(
                f"  Ex {i} (act={ex['max_activation']:.2f}, "
                f"peak={ex['peak_token']!r}): …{ctx}…"
            )
        return "\n".join(lines)
