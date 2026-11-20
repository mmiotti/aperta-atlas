"""Shared boilerplate for the per-variant PBF-stream scripts
(`networks_from_pbf`, `buildings_from_pbf`, `obstacles_from_pbf`, `pois_from_pbf`).

Each of those scripts goes through the same sequence: check `osmium` is on
PATH, resolve the source full-region PBF, clip it to the variant's
buffered AOI polygon via `osmium extract`, then stream-parse the clipped
PBF with a pyosmium handler. The handler implementation and per-handler
logging are the only things that differ between scripts — this module
factors out everything else.
"""
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

from shapely.geometry import MultiPolygon, Polygon

from aperta_atlas.context import Context, Storage
from aperta_atlas.utils import step

from preparation.world.common import buffered_place_polygon, write_poly_file


def _check_osmium_on_path() -> None:
    """Raise if `osmium` isn't on PATH (must be conda-installed)."""
    if shutil.which('osmium') is None:
        raise RuntimeError(
            "osmium-tool not found on PATH. Install via "
            "`conda install -c conda-forge osmium-tool`.")


def _source_pbf(context: Context, pbf_name: str) -> str:
    """Resolve the full-region source PBF path; raise if missing."""
    source_pbf = context.raw_path(
        Storage.PUBLIC, f'global/osm/{pbf_name}-latest.osm.pbf')
    if not os.path.exists(source_pbf):
        raise RuntimeError(
            f"Source PBF not found: {source_pbf}\n"
            f"Run `python -m preparation.world.osm.clip_pbf` first.")
    return source_pbf


@contextmanager
def clipped_pbf(context: Context, variant, out_name: str):
    """Yield `(clipped PBF path, buffered AOI polygon)` for `variant`, the PBF clipped by
    `osmium extract` inside a temp directory that is removed on exit. `out_name` is the
    basename of the temp files."""
    _check_osmium_on_path()
    source_pbf = _source_pbf(context, variant.pbf_name)

    with step('buffered_place_polygon'):
        polygon = buffered_place_polygon(variant.place, variant.buffer)

    with tempfile.TemporaryDirectory() as tmpdir:
        poly_path = os.path.join(tmpdir, f'{out_name}.poly')
        clipped = os.path.join(tmpdir, f'{out_name}_clipped.osm.pbf')

        write_poly_file(polygon, poly_path)

        with step('osmium extract (per-variant clip)'):
            subprocess.run(
                ['osmium', 'extract', '--polygon', poly_path,
                 source_pbf, '-o', clipped, '--overwrite'],
                check=True,
            )
            clip_mb = os.path.getsize(clipped) / (1024 * 1024)
            logging.info(f"  → clipped PBF: {clip_mb:,.1f} MB")

        yield clipped, polygon


def area_multipolygon(a) -> MultiPolygon | None:
    """Shapely multipolygon (lon/lat) of a pyosmium area: each outer ring with its inner
    rings as holes. None if no outer ring has at least 4 valid points."""
    polygons = []
    for outer in a.outer_rings():
        shell = [(n.lon, n.lat) for n in outer if n.location.valid()]
        if len(shell) < 4:
            continue
        holes = [[(n.lon, n.lat) for n in inner if n.location.valid()]
                 for inner in a.inner_rings(outer)]
        polygons.append(Polygon(shell, [h for h in holes if len(h) >= 4]))
    return MultiPolygon(polygons) if polygons else None
