"""
=============================================================================
Continuous value parser for L1000 shard post-processing
=============================================================================

Parses the text_data "dose" and "time" columns from existing shards,
extracts continuous float values, and writes them back into the shards.

New fields added to each shard:
  dose_um   float32 [B]  - dose converted to µM (NaN if unconvertible)
  dose_raw  float32 [B]  - raw numeric dose regardless of unit (NaN if missing)
  time_h    float32 [B]  - time in hours (NaN if missing)

=============================================================================
Usage
=============================================================================

python continuous_parse.py --shard_dir out_L1000/shards

This reads all shard_*.npz files, parses dose/time from text_data,
and overwrites the shards with the new continuous fields added.

=============================================================================
Dose conversion rules
=============================================================================

Concentration units (directly convertible to µM):
  uM (×1), µM (×1), nM (÷1000), pM (÷1e6), mM (×1000), M (×1e6)

Non-convertible (require molecular weight or different dimension):
  ng/ml, ug/ml, mg/ml, ng/uL → NaN (mass-per-volume, need MW)
  ng, ug           → NaN (mass, not concentration)
  uL, ml           → NaN (volume, not concentration)
  %                → NaN (percentage)
  Unknown, empty   → NaN

=============================================================================
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


# =============================================================================
# Dose unit conversion table
# =============================================================================

# Maps unit string (lowercase) → multiplier to convert to µM.
# None means conversion is not possible without additional info.
_DOSE_UNIT_TO_UM: Dict[str, Optional[float]] = {
    "um":    1.0,           # µM → µM (identity)
    "µm":    1.0,           # unicode µ variant
    "nm":    1e-3,          # nM → µM (÷1000)
    "pm":    1e-6,          # pM → µM (÷1e6)
    "mm":    1e3,           # mM → µM (×1000)
    "m":     1e6,           # M  → µM (×1e6)
    # ------
    "ng/ml": None,
    "ug/ml": None,
    "mg/ml": None,
    "ng/ul": None,
    "ng":    None,          
    "ug":    None,
    "ul":    None,          
    "ml":    None,
    "%":     None,          
    "unknown": None,
}


# =============================================================================
# Parsing helpers
# =============================================================================

# Regex to split "6.66 uM" → ("6.66", "uM")
_DOSE_RE = re.compile(r"^\s*([\d.eE+\-]+)\s+(.+?)\s*$")

# Regex to parse time like "24 h" or "72h"
_TIME_RE = re.compile(r"^\s*([\d.eE+\-]+)\s*h?\s*$", re.IGNORECASE)


def parse_dose_str(s: str) -> Tuple[float, str]:
    """
    Parse a dose string like "6.66 uM" into (value, unit).

    Returns (NaN, "") if parsing fails.
    """
    if not s or s.lower() in ("none", "nan", ""):
        return np.nan, ""
    m = _DOSE_RE.match(s)
    if m:
        try:
            val = float(m.group(1))
            unit = m.group(2).strip()
            return val, unit
        except ValueError:
            return np.nan, ""
    return np.nan, ""


def dose_to_um(value: float, unit: str) -> float:
    """Convert a dose value to µM given its unit string. Returns NaN if not convertible."""
    if np.isnan(value):
        return np.nan
    multiplier = _DOSE_UNIT_TO_UM.get(unit.lower())
    if multiplier is None:
        return np.nan
    return value * multiplier


def parse_time_str(s: str) -> float:
    """
    Parse a time string like "24 h" or "72h" into hours (float).

    Returns NaN if parsing fails.
    """
    if not s or s.lower() in ("none", "nan", ""):
        return np.nan
    m = _TIME_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    return np.nan


# =============================================================================
# Core: process one shard
# =============================================================================

def process_shard(path: Path) -> None:
    """
    Read a shard npz, parse dose/time from text_data, write back with
    dose_um, dose_raw, time_h added.
    """
    data = dict(np.load(str(path), allow_pickle=True))

    # ---- Locate dose and time columns in text_data ----
    if "text_cols" not in data or "text_data" not in data:
        print(f"[SKIP] {path.name}: no text_cols or text_data")
        return

    text_cols = list(data["text_cols"])
    text_data = data["text_data"]    # [B, C] unicode array
    B = text_data.shape[0]

    dose_idx = text_cols.index("dose") if "dose" in text_cols else None
    time_idx = text_cols.index("time") if "time" in text_cols else None

    # ---- Parse dose ----
    dose_um_arr = np.full(B, np.nan, dtype=np.float32)
    dose_raw_arr = np.full(B, np.nan, dtype=np.float32)

    if dose_idx is not None:
        for i in range(B):
            val, unit = parse_dose_str(str(text_data[i, dose_idx]))
            dose_raw_arr[i] = val
            dose_um_arr[i] = dose_to_um(val, unit)

    # ---- Parse time ----
    time_h_arr = np.full(B, np.nan, dtype=np.float32)

    if time_idx is not None:
        for i in range(B):
            time_h_arr[i] = parse_time_str(str(text_data[i, time_idx]))

    # ---- Write back ----
    data["dose_um"] = dose_um_arr
    data["dose_raw"] = dose_raw_arr
    data["time_h"] = time_h_arr

    np.savez_compressed(str(path), **data)


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Parse continuous dose/time values from L1000 shard text_data."
    )
    p.add_argument("--shard_dir", type=str, required=True,
                   help="Path to shards directory (e.g. out_L1000/shards)")
    args = p.parse_args()

    shard_dir = Path(args.shard_dir)
    shard_files = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_files:
        print(f"[ERROR] No shard_*.npz found in {shard_dir}")
        return

    print(f"[INFO] Found {len(shard_files)} shards in {shard_dir}")

    # ---- Stats ----
    total_samples = 0
    total_dose_um = 0
    total_dose_raw = 0
    total_time_h = 0
    unconvertible_units = set()

    for path in tqdm(shard_files, desc="[PARSE] Processing shards"):
        data = dict(np.load(str(path), allow_pickle=True))

        if "text_cols" not in data or "text_data" not in data:
            continue

        text_cols = list(data["text_cols"])
        text_data = data["text_data"]
        B = text_data.shape[0]
        total_samples += B

        dose_idx = text_cols.index("dose") if "dose" in text_cols else None

        # Parse and collect stats before writing
        dose_um_arr = np.full(B, np.nan, dtype=np.float32)
        dose_raw_arr = np.full(B, np.nan, dtype=np.float32)
        time_h_arr = np.full(B, np.nan, dtype=np.float32)

        if dose_idx is not None:
            for i in range(B):
                val, unit = parse_dose_str(str(text_data[i, dose_idx]))
                dose_raw_arr[i] = val
                um_val = dose_to_um(val, unit)
                dose_um_arr[i] = um_val
                if not np.isnan(val) and np.isnan(um_val) and unit:
                    unconvertible_units.add(unit)

        time_idx = text_cols.index("time") if "time" in text_cols else None
        if time_idx is not None:
            for i in range(B):
                time_h_arr[i] = parse_time_str(str(text_data[i, time_idx]))

        total_dose_um += np.count_nonzero(~np.isnan(dose_um_arr))
        total_dose_raw += np.count_nonzero(~np.isnan(dose_raw_arr))
        total_time_h += np.count_nonzero(~np.isnan(time_h_arr))

        # Write back
        data["dose_um"] = dose_um_arr
        data["dose_raw"] = dose_raw_arr
        data["time_h"] = time_h_arr
        np.savez_compressed(str(path), **data)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"[DONE] Processed {total_samples:,} samples across {len(shard_files)} shards")
    print(f"  dose_um  coverage: {total_dose_um:,}/{total_samples:,} "
          f"({total_dose_um / total_samples * 100:.1f}%)" if total_samples > 0 else "")
    print(f"  dose_raw coverage: {total_dose_raw:,}/{total_samples:,} "
          f"({total_dose_raw / total_samples * 100:.1f}%)" if total_samples > 0 else "")
    print(f"  time_h   coverage: {total_time_h:,}/{total_samples:,} "
          f"({total_time_h / total_samples * 100:.1f}%)" if total_samples > 0 else "")
    if unconvertible_units:
        print(f"  Unconvertible dose units: {sorted(unconvertible_units)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
