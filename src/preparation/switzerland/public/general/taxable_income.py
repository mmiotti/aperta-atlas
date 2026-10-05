"""
Per-municipality average taxable income (CHF), from BFS public tax statistics.

Joins the source `Steuerbares Einkommen pro Einwohner/-in` series onto the existing
`municipalities` table (created by `political_boundaries.py`) and writes a companion
`municipalities_income.csv` in the same namespace.

Inputs (under <DATA_DIR_PUBLIC>/raw/switzerland/income/):
    KM10-27600-18-c-polg-2024-d-APPENDIX/27600_DE.csv
Inputs (same-namespace; prior stage):
    municipalities.csv  (from political_boundaries.py)

Outputs (PUBLIC, under preparation/switzerland/general/):
    municipalities_income.csv

Run:
    python -m preparation.switzerland.public.general.taxable_income
"""

import logging

import pandas as pd

from aperta_atlas.context import init_context
from aperta_atlas.context import Storage


_VARIABLE_NAME = 'Steuerbares Einkommen pro Einwohner/-in, in Franken'
_RAW_PATH = 'switzerland/income/KM10-27600-18-c-polg-2024-d-APPENDIX/27600_DE.csv'


def main():
    context = init_context()
    raw = pd.read_csv(context.raw_path(Storage.PUBLIC, _RAW_PATH), sep=';', index_col=0)
    raw = (raw[raw['VARIABLE'] == _VARIABLE_NAME][['GEO_NAME', 'VALUE']]
           .rename(columns={'VALUE': 'taxable_income'})
           .set_index('GEO_NAME'))

    municipalities = context.get_generic('municipalities.csv')
    municipalities = municipalities.join(raw, how='left')
    n_missing = raw.index.difference(municipalities.index).size
    if n_missing:
        logging.warning(
            f"{n_missing:,} taxable-income rows did not match any municipality "
            f"(by GEO_NAME).")

    context.create_generic(municipalities[['taxable_income']], 'municipalities_income.csv')
    context.close()


if __name__ == '__main__':
    main()
