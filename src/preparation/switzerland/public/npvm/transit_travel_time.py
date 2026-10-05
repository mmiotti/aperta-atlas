"""
Extract zone-to-zone public-transit travel times from the NPVM 2023 .mtx file.

Reads ARE's NPVM (Nationales Personenverkehrsmodell) 2023 transit travel-time
matrix — a VISUM `$O;D3` text-format file, ~2.4 GB, ~63M OD pairs over
~7900 zones. Values are total trip time (first-mile access + in-vehicle
+ wait + transfer + last-mile egress, in minutes).

Output: an aperta `TieredODGeoPairs` with the `zones_to_zones` tier
populated. Zone IDs match the prefix scheme from `traffic_zones.py`:
`ZD<n>` for domestic, `ZF<n>` for foreign. Values are converted from
minutes to seconds (aperta-wide convention for travel-time ODMs).

Pairs whose origin OR destination NPVM zone isn't present in our
prepared `traffic_zones.csv` are dropped — those are usually internal
NPVM artefacts (centroid-of-centroid placeholders, etc.) not relevant to
downstream accessibility work.

Inputs (PUBLIC, under <DATA_DIR_PUBLIC>/raw/switzerland/npvm/2023/transit/):
    RITA_ACT_EGT_NPVM_2023.mtx                 # VISUM $O;D3 text matrix

Inputs (same-namespace; prior stage):
    traffic_zones.csv (from general/traffic_zones.py)    # zone_id prefix scheme

Outputs (PUBLIC, under preparation/switzerland/npvm/):
    odm/npvm_2023_transit_idx.npz              # zones_to_zones: dest IDs per origin
    odm/npvm_2023_transit_travel_time.npz      # zones_to_zones: seconds per OD pair

Run:
    python -m preparation.switzerland.public.npvm.transit_travel_time
"""

import logging

import numpy as np
import pandas as pd

from aperta.od_pairs import TieredODGeoPairs
from aperta_atlas.context import Storage, init_context
from aperta_atlas.utils import step


# The .mtx file is one metric per file with a fixed 8-line header. The body
# starts at line 9; each line is `<orig_npvm_id> <dest_npvm_id> <value>`
# (whitespace-separated). Encoding is UTF-8 with BOM. Values are in
# MINUTES (we convert to seconds on save).
_MTX_HEADER_LINES = 8
_MTX_REL = 'switzerland/npvm/2023/transit/RITA_ACT_EGT_NPVM_2023.mtx'


def _parse_factor_line(path: str) -> float:
    """Read the scaling factor from line 5 of an NPVM .mtx header.

    The 8-line header is:
        1: $O;D3   (format directive, after UTF-8 BOM)
        2: * Von Bis
        3: - -
        4: * Faktor
        5: <factor>   ← this line
        6: *
        7: * <agency>
        8: * <date>
    """
    with open(path, encoding='utf-8-sig') as f:
        for _ in range(5):
            line = f.readline()
        return float(line.strip())


def _count_data_rows(path: str) -> int:
    """Count data rows between the 8-line header and the trailing names
    section (marked by `* Netzobjektnamen` followed by `$NAMES`). Uses
    `mmap` so the scan runs in ~5 s on a 2.4 GB file.

    Raises ValueError if the separator marker isn't found — that would
    indicate a format change worth flagging rather than silently reading
    past it.
    """
    import mmap
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
        # Search for the newline immediately before the `* Netzobjektnamen`
        # separator. Robust to `\n` and `\r\n` line endings (in either case
        # the byte right before `* Netzobjektnamen` is `\n`).
        sep_idx = mm.find(b'\n* Netzobjektnamen')
        if sep_idx < 0:
            # Fallback: try locating `$NAMES` directly. Same logic.
            sep_idx = mm.find(b'\n$NAMES')
            if sep_idx < 0:
                raise ValueError(
                    "Neither `* Netzobjektnamen` nor `$NAMES` marker found "
                    "in .mtx file — VISUM $O format may have changed.")
        # Count newlines in [0, sep_idx + 1) via chunked reads — bytes.count
        # is C-fast and the chunk size bounds peak memory. The number of
        # newlines equals the number of lines ending at or before sep_idx,
        # which is the header + data line count combined.
        mm.seek(0)
        remaining = sep_idx + 1
        chunk_bytes = 64 * 1024 * 1024
        lines_before_sep = 0
        while remaining > 0:
            chunk = mm.read(min(chunk_bytes, remaining))
            lines_before_sep += chunk.count(b'\n')
            remaining -= len(chunk)
        return lines_before_sep - _MTX_HEADER_LINES


def read_npvm_mtx(path: str) -> pd.DataFrame:
    """Parse one VISUM `$O;D3` text matrix into a long-form DataFrame.

    Returns a DataFrame with columns `orig`, `dest`, `value` (int64, int64,
    float64). Applies the file's header `Faktor` to `value` if != 1.0.

    The data section is bounded by `_count_data_rows()` (mmap-based scan
    for the trailing `$NAMES` block) and passed to pandas as `nrows=`, so
    the per-zone name listing at the end of the file is skipped cleanly
    without comment/bad-line handling.
    """
    factor = _parse_factor_line(path)
    n_data_rows = _count_data_rows(path)
    df = pd.read_csv(
        path,
        sep=r'\s+',
        skiprows=_MTX_HEADER_LINES,
        nrows=n_data_rows,
        header=None,
        names=['orig', 'dest', 'value'],
        encoding='utf-8-sig',
        dtype={'orig': 'int64', 'dest': 'int64', 'value': 'float64'},
        engine='c',
    )
    if factor != 1.0:
        df['value'] = df['value'] * factor
        logging.info(f"  → applied scaling factor {factor}")
    return df


def main():
    context = init_context()

    with step('load prepared traffic zones (for ZD/ZF zone-id mapping)'):
        zones = context.source('preparation/switzerland/general').get_generic(
            'traffic_zones.csv')
        # `zones.index` is the prefixed string (`ZD<n>` / `ZF<n>`); recover
        # the numeric NPVM id by stripping the 2-char prefix.
        npvm_id = (
            zones.index.to_series()
            .str.removeprefix('ZD').str.removeprefix('ZF')
            .astype(int)
        )
        npvm_to_zone_id: dict[int, str] = dict(zip(npvm_id, zones.index))
        n_dom = int(zones['is_domestic'].sum())
        logging.info(
            f"  → {len(zones):,} zones ({n_dom:,} domestic / "
            f"{len(zones) - n_dom:,} foreign)")

    with step('parse NPVM 2023 transit .mtx (2.4 GB, ~63M OD pairs)'):
        in_path = context.raw_path(Storage.PUBLIC, _MTX_REL)
        df = read_npvm_mtx(in_path)
        logging.info(f"  → {len(df):,} OD pairs read")

    with step('map NPVM int IDs → ZD/ZF prefixed zone IDs'):
        df['orig_id'] = df['orig'].map(npvm_to_zone_id)
        df['dest_id'] = df['dest'].map(npvm_to_zone_id)
        n_before = len(df)
        df = df.dropna(subset=['orig_id', 'dest_id'])
        logging.info(
            f"  → {len(df):,}/{n_before:,} pairs kept "
            f"({n_before - len(df):,} dropped: zone not in prepared "
            f"traffic_zones.csv)")

    with step('NaN out unreachable-pair sentinels (value > 24 h in minutes)'):
        # VISUM encodes "no route" with a large integer stub (~3e6 min
        # observed for NPVM 2023 transit). Anything above 24 h is not a
        # real PT trip. Set to NaN (aperta convention for unreachable
        # destinations — routing under a cutoff produces NaN too), so the
        # (orig, dest) pair still appears in the ODM and downstream code
        # skips it uniformly.
        sentinel_mask = df['value'] > 24 * 60
        n_sentinel = int(sentinel_mask.sum())
        df.loc[sentinel_mask, 'value'] = float('nan')
        logging.info(
            f"  → NaN'd {n_sentinel:,}/{len(df):,} pairs "
            f"({100*n_sentinel/max(len(df), 1):.2f} %) as unreachable sentinels")

    with step('convert minutes → seconds, group into per-origin arrays'):
        # NPVM-published transit times are minutes; aperta-wide convention
        # for travel-time ODMs is seconds.
        df['value_s'] = (df['value'] * 60.0).astype(np.float32)
        # Canonical ordering: sort by (origin, destination) so idx and values
        # are aligned and reproducible across runs.
        df = df.sort_values(['orig_id', 'dest_id'], kind='stable')
        zones_idx: dict[str, list[str]] = {}
        zones_values: dict[str, np.ndarray] = {}
        for orig_id, group in df.groupby('orig_id', sort=False):
            zones_idx[orig_id] = group['dest_id'].tolist()
            zones_values[orig_id] = group['value_s'].to_numpy()
        n_origins = len(zones_idx)
        median_t = float(df['value_s'].median())
        logging.info(
            f"  → {n_origins:,} origin zones; median t = "
            f"{median_t/60:.1f} min ({median_t:.0f} s)")

    with step('save ODM (idx + travel_time)'):
        idx_pairs = TieredODGeoPairs(zones_to_zones=zones_idx)
        val_pairs = TieredODGeoPairs(zones_to_zones=zones_values)
        context.create_tiered_odm(
            idx_pairs, network_name='npvm_2023_transit', data_name='idx')
        context.create_tiered_odm(
            val_pairs, network_name='npvm_2023_transit', data_name='travel_time')

    context.close()


if __name__ == '__main__':
    main()
