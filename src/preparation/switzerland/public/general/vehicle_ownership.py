"""
Per-municipality vehicle counts by type and fuel, plus per-capita and share columns,
from BFS open-data PX-Cube `px-x-1103020100_111.px`.

Pivots the (vehicle_type, fuel_type) → count table for one reference year, joins it
to the `municipalities` properties via `bfs_number`, and derives:
    * `<vehicle_type>_all`   — sum over all fuel types
    * `<vehicle_type>_main`  — sum over `MAIN_FUEL_TYPES` only
    * `veh_per_capita_<vehicle_type>` — divided by population
    * `share_all_<vehicle_type>_<fuel_type>`  — share of total
    * `share_main_<vehicle_type>_<fuel_type>` — share of `MAIN_FUEL_TYPES` sum (only
      for fuel types in `MAIN_FUEL_TYPES`)

Inputs (under <DATA_DIR_PUBLIC>/raw/switzerland/vehicle_ownership/):
    px-x-1103020100_111.px
Inputs (same-namespace; prior stage):
    municipalities.csv  (from political_boundaries.py)

Outputs (PUBLIC, under preparation/switzerland/general/):
    municipalities_car_ownership.csv

Run:
    python -m preparation.switzerland.public.general.vehicle_ownership
"""

import re

import pandas as pd
from pyaxis import pyaxis

from aperta_atlas.context import init_context
from aperta_atlas.context import Storage


# Reference year for the snapshot. Update when a new vintage is preferred.
DATA_YEAR = 2023

VEHICLE_NAMES: dict[str, str] = {
    'Anhänger':                       'trailers',
    'Industriefahrzeuge':             'industry',
    'Landwirtschaftsfahrzeuge':       'agriculture',
    'Motorräder':                     'motorcycles',
    'Personentransportfahrzeuge':     'passenger_transport',
    'Personenwagen':                  'cars',
    'Sachentransportfahrzeuge':       'cargo_transport',
}

FUEL_TYPE_NAMES: dict[str, str] = {
    'Benzin':                              'icev',
    'Diesel':                              'icev_diesel',
    'Benzin-elektrisch: Normal-Hybrid':    'hev',
    'Benzin-elektrisch: Plug-in-Hybrid':   'phev',
    'Diesel-elektrisch: Normal-Hybrid':    'hev_diesel',   # ~18% of HEVs
    'Diesel-elektrisch: Plug-in-Hybrid':   'phev_diesel',  # ~3% of PHEVs
    'Elektrisch':                          'bev',
    'Wasserstoff':                         'fcv',
    'Gas (mono- und bivalent)':            'other',
    'Anderer':                             'other',
    'Ohne Motor':                          'none',
}

# Fuel types tracked under `_main` shares (drops diesel-hybrid sub-categories etc.).
MAIN_FUEL_TYPES = ('icev', 'icev_diesel', 'hev', 'phev', 'bev')

_RAW_PATH = 'switzerland/vehicle_ownership/px-x-1103020100_111.px'


def get_bfs_number_and_name(s: str) -> tuple[int, str]:
    """Split a BFS string like '1234 Gemeindename' into `(1234, 'Gemeindename')`."""
    m = re.match(r'([0-9]+) (.+)', s)
    if m is None:
        raise ValueError(f"Could not parse BFS row label {s!r}.")
    return int(m.group(1)), m.group(2)


def main():
    context = init_context()

    px = pyaxis.parse(uri=context.raw_path(Storage.PUBLIC, _RAW_PATH), encoding='UTF-8')
    df = px['DATA']
    df['DATA'] = df['DATA'].astype(int)
    df = df[df['Jahr'] == str(DATA_YEAR)].copy()
    df['Fahrzeuggruppe'] = df['Fahrzeuggruppe'].replace(VEHICLE_NAMES)
    df['Treibstoff'] = df['Treibstoff'].replace(FUEL_TYPE_NAMES)
    df = df.rename(columns={'Fahrzeuggruppe': 'vehicle_type', 'Treibstoff': 'fuel_type'})
    vehicle_types = df['vehicle_type'].unique()
    fuel_types = df['fuel_type'].unique()

    # Pivot to one column per (vehicle_type, fuel_type) pair, then flatten the
    # MultiIndex header into single 'vehicle_fuel' column names.
    counts = (df.groupby(['Gemeinde', 'vehicle_type', 'fuel_type'])
                .agg({'DATA': 'sum'})
                .unstack(level=[1, 2]))
    counts.columns = counts.columns.droplevel()
    counts = counts.reset_index()
    bfs = pd.DataFrame.from_records(
        counts['Gemeinde'].apply(get_bfs_number_and_name),
        index=counts.index,
        columns=pd.MultiIndex.from_product([['municipality'], ['bfs_number', 'name']]),
    )
    counts = (counts.join(bfs, how='left')
                    .set_index([('municipality', 'bfs_number')])
                    .drop(columns=[('Gemeinde', '')]))
    counts.columns = ['_'.join(idx) for idx in counts.columns.to_flat_index()]

    municipalities = context.get_generic('municipalities.csv')
    municipalities = municipalities.join(counts, how='left', on='bfs_number')

    for vehicle_type in vehicle_types:
        veh_cols = [c for c in counts.columns if c.startswith(vehicle_type)]
        main_cols = [f'{vehicle_type}_{ft}' for ft in MAIN_FUEL_TYPES]
        municipalities[f'{vehicle_type}_all'] = municipalities[veh_cols].sum(axis=1)
        municipalities[f'{vehicle_type}_main'] = municipalities[main_cols].sum(axis=1)
        municipalities[f'veh_per_capita_{vehicle_type}'] = (
            municipalities[f'{vehicle_type}_all'] / municipalities['population']
        )
        for fuel_type in fuel_types:
            municipalities[f'share_all_{vehicle_type}_{fuel_type}'] = (
                municipalities[f'{vehicle_type}_{fuel_type}']
                / municipalities[f'{vehicle_type}_all']
            )
            if fuel_type in MAIN_FUEL_TYPES:
                municipalities[f'share_main_{vehicle_type}_{fuel_type}'] = (
                    municipalities[f'{vehicle_type}_{fuel_type}']
                    / municipalities[f'{vehicle_type}_main']
                )

    # TODO: limit veh_per_capita_cars to >0 and <=1.3. One (?) municipality has 0,
    # one has very high number (up to 8.9).
    out_cols = [c for c in municipalities.columns
                if c.startswith(('share_', 'veh_per_capita_'))]
    context.create_generic(municipalities[out_cols], 'municipalities_car_ownership.csv')
    context.close()


if __name__ == '__main__':
    main()
