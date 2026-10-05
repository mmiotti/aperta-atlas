"""
Centralised analysis-area definitions for the `preparation/world/` pipeline.

An **`Area`** is one analysis target — a country, canton, city, or any
OSM-geocodable place — plus all the per-area configuration that's
shared across multiple preparation scripts (the OSM place name, the
filesystem identifier, the per-mode network buffers, etc.).

Each script in `preparation/world/{osm,elevation,land_use}/` consumes
`AREAS` and fans it out into its own Variants. Adding a new entry here
makes it available in every script automatically; tweaking a buffer
for an existing area propagates everywhere.

The split between area-level and script-level configuration:

- **Area-level** (lives here): `place`, `name`, `source_pbf`,
  per-mode `buffers`, `ghsl_tiles` (for the population script),
  `aoi_buffer_m`. Anything that should be consistent across multiple
  scripts for the same analysis area.

- **Script-level** (lives in each script as module-level dicts):
  per-mode tuning like `network_type`, `prune_dead_end_max_m`,
  `tolerance_m`, `obstacle_buffer_m`. These are script-specific and
  don't cross over.

For mode-agnostic data (POIs, obstacles, buildings, DEM, population)
the buffer should be the widest across modes — `widest_buffer(area)`
returns this.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Area:
    """One analysis area + the per-area configuration shared across
    preparation scripts."""

    name: str
    """Short filesystem-safe identifier. Used everywhere the area
    appears as a filename anchor or variant-name prefix (e.g.
    `switzerland`, `bern`). Lowercase, no spaces."""

    place: str | tuple[str, ...]
    """OSM-geocodable name (or tuple of names) passed to `ox.geocode_to_gdf`.
    Follows OSM nominatim conventions:
        'Switzerland', 'Canton of Bern, Switzerland', 'Zürich, Switzerland'.

    A tuple is geocoded piecewise and unioned before buffering — used
    by multi-region Areas such as language-group cross-validation
    scenarios where the AOI is a union of cantons."""

    source_pbf: str
    """Anchor for the source Geofabrik PBF filename (without the
    `-latest.osm.pbf` suffix). E.g. `'europe'` → consumes
    `raw/global/osm/europe-latest.osm.pbf`. The clip step
    (`clip_pbf.py`) reads this; downstream scripts reference the
    *clipped* output via `f'{area.name}_buffered'` instead."""

    buffers: dict[str, int]
    """Per-mode buffer in metres. Modes are typically `'walk'`,
    `'bike'`, `'car'`. Each mode's buffer is its cross-border
    accessibility envelope — how far outside the AOI (area polygon +
    `aoi_buffer_m`) edges can extend so border-region routing stays
    correct."""

    ghsl_tiles: tuple[str, ...]
    """GHS-POP R2023A 100 m tile IDs covering the buffered area
    polygon. Read by `land_use/population_ghs.py`. Switzerland sits in
    `R4_C19` (10° × 10° tile in Mollweide). Larger areas may need
    multiple tiles."""

    aoi_buffer_m: int = 0
    """Buffer (m) added to the OSM `place` polygon to define the AOI.
    For small city-level areas the un-buffered polygon (e.g. City of
    Cambridge ~41 km²) excludes the immediate surroundings — set this
    to widen the "analysis area" without changing the OSM place name.
    Defaults to 0 (use the place polygon exactly). The per-mode
    `buffers` are added ON TOP OF this for the data-extent envelope —
    `widest_buffer(area)` returns `aoi_buffer_m + max(buffers.values())`
    so preparation scripts automatically scope their data to cover the
    full AOI + cross-border ring."""

    is_swiss: bool = False
    """True iff the area lies entirely within Switzerland, in which case
    Swiss-specific data sources (STATPOP / STATENT hectares, MTMC / MOBIS
    survey microdata, NPVM traffic zones) are available for its extent.
    Consulted by scripts that need one of those inputs (e.g.
    `population_per_building_from_coef` for STATPOP density
    stratification; `Scenario.__post_init__` for `cell_source='hectares'`
    validation). Set to False (default) for non-Swiss areas."""


AREAS: dict[str, Area] = {
    'switzerland': Area(
        name='switzerland',
        place='Switzerland',
        source_pbf='europe',
        buffers={'walk': 5_000, 'bike': 25_000, 'car': 50_000},
        ghsl_tiles=('R4_C19',),
        is_swiss=True,
    ),
    'bern': Area(
        name='bern',
        place='Canton of Bern, Switzerland',
        source_pbf='europe',
        buffers={'walk': 5_000, 'bike': 20_000, 'car': 40_000},
        ghsl_tiles=('R4_C19',),
        is_swiss=True,
    ),
    # Tighter Bern-area entry used by the Technical Validation scenarios
    # (`scenarios_technical_validation.py`). Verwaltungskreis Bern-Mittelland
    # is the administrative district surrounding Bern city (~940 km²,
    # ~415k pop). Chosen as a close proxy for BFS's "Agglomeration Bern"
    # (~140 communes, ~410k pop) — same population footprint, but an
    # OSM-geocodable name. Buffers sized so a ~20 min radius and/or k=10
    # destinations are captured for all modes.
    'bern-metro': Area(
        name='bern-metro',
        place='Bern-Mittelland, Switzerland',
        source_pbf='europe',
        buffers={'walk': 5_000, 'bike': 10_000, 'car': 20_000},
        ghsl_tiles=('R4_C19',),
        is_swiss=True,
    ),
    'cambridgeuk': Area(
        name='cambridgeuk',
        place='Cambridge, Cambridgeshire, England',
        source_pbf='europe',
        buffers={'walk': 5_000, 'bike': 20_000, 'car': 40_000},
        # Make area of interest (AOI) a bit larger than city itself.
        aoi_buffer_m=3_000,
        # Cambridge + 40 km buffer straddles the prime meridian, so the
        # buffered AOI spans two adjacent 10° GHSL tiles.
        ghsl_tiles=('R3_C18', 'R3_C19'),
    ),
    # Language-region areas used by the spatial cross-validation scenarios
    # (`scenarios_technical_validation.py`, `cv-*`). Multilingual cantons
    # (BE, FR, VS, GR) are excluded from BOTH so training and test
    # populations are linguistically homogeneous. Buffers match the tv-
    # scenarios (bern-metro sizing) to keep the network small; cross-
    # border routing across the excluded ring is accepted as a trade-off.
    'switzerland-de': Area(
        name='switzerland-de',
        # 17 homogeneous German-speaking cantons (BE excluded — bilingual).
        # Native "Kanton X" naming — English "Canton of X" made Nominatim
        # sometimes return the city center point (e.g. Zurich, St. Gallen)
        # before the canton polygon, which `osmnx._get_first_polygon` rejects.
        place=(
            'Kanton Zürich, Switzerland',
            'Kanton Basel-Stadt, Switzerland',
            'Kanton Basel-Landschaft, Switzerland',
            'Kanton Aargau, Switzerland',
            'Kanton Solothurn, Switzerland',
            'Kanton Luzern, Switzerland',
            'Kanton Zug, Switzerland',
            'Kanton Schwyz, Switzerland',
            'Kanton Uri, Switzerland',
            'Kanton Nidwalden, Switzerland',
            'Kanton Obwalden, Switzerland',
            'Kanton Glarus, Switzerland',
            'Kanton St. Gallen, Switzerland',
            'Kanton Appenzell Ausserrhoden, Switzerland',
            'Kanton Appenzell Innerrhoden, Switzerland',
            'Kanton Thurgau, Switzerland',
            'Kanton Schaffhausen, Switzerland',
        ),
        source_pbf='europe',
        buffers={'walk': 5_000, 'bike': 10_000, 'car': 20_000},
        ghsl_tiles=('R4_C19',),
        is_swiss=True,
    ),
    'switzerland-fr': Area(
        name='switzerland-fr',
        # 4 French-speaking cantons (VS/FR/BE excluded — bilingual FR/DE).
        # Contiguous: JU shares a direct border with NE, so no gap needs
        # to be bridged by the buffer. Ticino was previously included but
        # dropped because it's an isolated Italian-speaking region
        # separated from the FR cluster by VS + GR (both excluded), with
        # no buffer wide enough to bridge — a two-piece AOI would break
        # network routing between the pieces.
        # Native "Canton de X" naming — English "Canton of X" made
        # Nominatim sometimes return the city center point before the
        # canton polygon (see switzerland-de note above).
        place=(
            'Canton de Genève, Switzerland',
            'Canton de Vaud, Switzerland',
            'Canton de Neuchâtel, Switzerland',
            'Canton du Jura, Switzerland',
        ),
        source_pbf='europe',
        buffers={'walk': 5_000, 'bike': 10_000, 'car': 20_000},
        ghsl_tiles=('R4_C19',),
        is_swiss=True,
    ),
}


def widest_buffer(area: Area) -> int:
    """Largest buffer (m) FROM `area.place` for mode-agnostic data
    (POIs, obstacles, buildings, DEM, population).

    Composed of two pieces:
      - `area.aoi_buffer_m` — extends the AOI itself beyond the raw
        OSM polygon (useful for small city cases).
      - `max(area.buffers.values())` — the cross-border ring on TOP of
        the AOI for any mode's accessibility radius.

    Using a single combined buffer here keeps the file count small
    (no `<area>_pois_walk.gpkg` / `<area>_pois_bike.gpkg`
    proliferation) at the cost of slightly over-fetching for
    narrower-mode use.
    """
    return area.aoi_buffer_m + max(area.buffers.values())
