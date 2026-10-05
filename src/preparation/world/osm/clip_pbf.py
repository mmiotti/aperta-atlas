"""
Clip a large source OSM PBF (continent / multi-country) down to a
buffered AOI polygon for the current area. Run before any downstream
`preparation/world/osm/` script.

Why osmium-tool rather than Python: osmium streams the input PBF and
clips with bounded RAM (hundreds of MB regardless of input size).
Equivalent Python clipping would not work on Europe-scale PBFs.

Cases are defined in `preparation/world/areas.py`; this script registers
one `<area_name>_buffered` variant per area. The buffer is the widest
per-mode buffer of the area so every downstream per-mode variant fits
inside the clipped PBF.

Inputs (PUBLIC, under raw/global/osm/):
    <source_pbf>-latest.osm.pbf         # e.g. europe-latest.osm.pbf
                                        # https://download.geofabrik.de/

Outputs (PUBLIC, under raw/global/osm/):
    <area_name>_buffered.poly             # OSM .poly polygon (osmium input)
    <area_name>_buffered-latest.osm.pbf   # spatially clipped PBF

Requires `osmium-tool` on PATH (`conda install -c conda-forge osmium-tool` or
`brew install osmium-tool`).

Run all variants sequentially (default):
    python -m preparation.world.osm.clip_pbf
Single variant:
    python -m preparation.world.osm.clip_pbf --variant switzerland_buffered
    python -m preparation.world.osm.clip_pbf --variant bern_buffered
"""

import os
import shutil
import subprocess

from aperta_atlas.context import init_context, Storage
from aperta_atlas.variant import Variants

from preparation.world.areas import AREAS, widest_buffer
from preparation.world.common import buffered_place_polygon, write_poly_file


# `buffer` here is the widest per-mode buffer for the area — the clip
# step must cover any downstream per-mode variant. `area_name` becomes
# the filename anchor for the output `<area_name>_buffered-latest.osm.pbf`.
variants = Variants([
    ('place', str), ('area_name', str), ('source_pbf', str), ('buffer', int),
])
for area in AREAS.values():
    variants.add(
        name=f'{area.name}_buffered',
        place=area.place, area_name=area.name,
        source_pbf=area.source_pbf, buffer=widest_buffer(area),
    )


def main(variant) -> None:
    if shutil.which('osmium') is None:
        raise RuntimeError(
            "osmium-tool not found on PATH. Install via "
            "`conda install -c conda-forge osmium-tool` or "
            "`brew install osmium-tool`.")

    context = init_context(variant)
    name = f'{variant.area_name}_buffered'
    poly_path = context.raw_path(Storage.PUBLIC, f'global/osm/{name}.poly')
    source_pbf = context.raw_path(Storage.PUBLIC, f'global/osm/{variant.source_pbf}-latest.osm.pbf')
    output_pbf = context.raw_path(Storage.PUBLIC, f'global/osm/{name}-latest.osm.pbf')

    if not os.path.exists(source_pbf):
        raise RuntimeError(
            f"Source PBF not found: {source_pbf}\n"
            f"Download from https://download.geofabrik.de/ and place at "
            f"this path before re-running.")

    # 1. Build buffered polygon, write to OSM .poly format.
    polygon = buffered_place_polygon(variant.place, variant.buffer)
    write_poly_file(polygon, poly_path)

    # 2. osmium extract — spatial clip via .poly. Streams source PBF,
    #    bounded RAM (~few hundred MB regardless of input size).
    subprocess.run(
        ['osmium', 'extract', '--polygon', poly_path,
         source_pbf, '-o', output_pbf, '--overwrite'],
        check=True,
    )

    context.close()


if __name__ == '__main__':
    variants.run(main)
