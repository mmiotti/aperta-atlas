"""
Engine behind `a_mzmv_standardize.py`: reads one survey year's raw files, decodes them with that
year's codebook and brings the result onto the canonical schema (`schema.py`).

A codebook module (`codebook_<year>.py`) defines:

    YEAR        survey year
    RAW_DIR     folder under raw/switzerland/mzmv/
    RAW_FILES   {alias: file stem}; matched case-insensitively, any extension
    READ_KWS    `pandas.read_csv` keywords (separator, encoding)
    XY_CRS      CRS of the raw coordinates, or None to locate by postcode / municipality
    XY_CRS_BY_TABLE  optional {table: CRS} for tables whose coordinates differ (2005 journeys)
    households(raw), household_members(raw), people(raw), vehicles(raw), trips(raw), legs(raw)
                one function per table, taking {alias: raw DataFrame with lower-case
                column names} and returning the table's canonical columns (any subset;
                the rest are added empty) plus raw keys the engine turns into ids:
                `hhnr`; `person_no` (people, trips, legs, journeys; members where the data
                says); `member_no`; `vehicle_type` + `vehicle_no`; `trip_no`; `leg_no`;
                `journey_no`; `main_driver_nos` (vehicles); `car_no` (legs: n-th household
                car); and `<prefix>_x/_y` in XY_CRS where the year has coordinates.
    day_journeys(raw), overnight_journeys(raw), journey_places(raw)
                the same, for years with the journey modules (2005 on), with `place_type`
                + `place_no` for the places; without them the tables are written empty.

The decoding helpers (`num`, `coord`, `text`, `cat`, `flag`, `any_flag`, `bands`, `numbers`,
`hhmm_to_min`, `parse_date`) are what the codebooks use. `cat` and `flag` log raw codes the
codebook does not list, so a first run of a year shows where the documentation and the data
disagree.
"""

import logging
import os
from types import ModuleType
from typing import Iterable

import numpy as np
import pandas as pd

from aperta.errors import DataError
from aperta_atlas.context import Context, Storage

from preparation.switzerland.private.surveys.mzmv import geo, schema


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_raw(context: Context, codebook: ModuleType) -> dict[str, pd.DataFrame]:
    """{alias: raw table} for one year, with stripped lower-case column names."""
    raw_dir = context.raw_path(Storage.PRIVATE, f'switzerland/mzmv/{codebook.RAW_DIR}')
    if not os.path.isdir(raw_dir):
        raise DataError(f"{codebook.YEAR}: raw folder not found: {raw_dir}")
    files = {os.path.splitext(f)[0].lower(): f
             for f in os.listdir(raw_dir) if not f.startswith('.')}
    raw = {}
    for alias, stem in codebook.RAW_FILES.items():
        name = files.get(stem.lower())
        if name is None:
            raise DataError(
                f"{codebook.YEAR}: no raw file '{stem}.*' in {raw_dir} (found: "
                f"{sorted(files.values())}). Fix RAW_FILES in codebook_{codebook.YEAR}.py.")
        kws = dict(codebook.READ_KWS)
        if kws.get('engine') != 'python':
            kws.setdefault('low_memory', False)     # one dtype per column, no DtypeWarning
        df = pd.read_csv(os.path.join(raw_dir, name), **kws)
        if df.shape[1] < 2:
            raise DataError(f"{name} parsed to {df.shape[1]} column(s): wrong separator? "
                            f"Fix READ_KWS in codebook_{codebook.YEAR}.py.")
        df.columns = df.columns.str.strip().str.lower()
        if df.columns.duplicated().any():
            raise DataError(f"{name}: column names clash when lower-cased: "
                            f"{sorted(set(df.columns[df.columns.duplicated()]))}")
        logging.info(f"  {name}: {len(df):,} rows, {df.shape[1]} columns")
        raw[alias] = df
    return raw


# ---------------------------------------------------------------------------
# Decoding helpers for the codebooks
# ---------------------------------------------------------------------------

def _to_float(s: pd.Series) -> pd.Series:
    """`s` as float; text is parsed with ',' or '.' as decimal mark. Empty strings and the
    SPSS system-missing '.' become NaN silently, anything else unparsable with a warning."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype('string').str.strip()
    txt = txt.mask(txt.isin(['', '.']).fillna(False))
    out = pd.to_numeric(txt.str.replace(',', '.', regex=False), errors='coerce').astype(float)
    bad = out.isna() & txt.notna()
    if bad.any():
        logging.warning(f"  {s.name}: {int(bad.sum()):,} non-numeric values set to NaN, "
                        f"e.g. {txt[bad].unique()[:5].tolist()}")
    return out


def num(s: pd.Series, na: Iterable[float] = (), allow_negative: bool = False) -> pd.Series:
    """`s` as float. Negative values (the BFS missing codes from 2000 on) and the codes in
    `na` (1994 used e.g. 8 / 9 for don't know / no answer) become NaN."""
    x = _to_float(s)
    drop = x.isin(list(na))
    if not allow_negative:
        drop |= x < 0
    return x.mask(drop)


def coord(s: pd.Series) -> pd.Series:
    """Raw coordinate as float, with the BFS missing codes as NaN. Unlike `num`, negative
    values are kept (WGS84 longitudes west of Greenwich)."""
    return num(s, na=(-97, -98, -99, -997, -999), allow_negative=True)


def text(s: pd.Series) -> pd.Series:
    """Raw text codes (e.g. registry codes) as stripped strings; empty and BFS missing codes
    become NaN, and a trailing '.0' from numbers read as floats is dropped."""
    t = s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    return t.mask(t.isin(['', '.', '-97', '-98', '-99']).fillna(False)).astype(object)


def numbers(*cols: pd.Series, na: Iterable[float] = ()) -> pd.Series:
    """Positive whole numbers from several columns (e.g. the member numbers of a car's main
    drivers), `;`-joined per row; NaN where there are none. Codes in `na` are dropped."""
    df = pd.concat([num(c, na=na) for c in cols], axis=1)
    return df.apply(lambda r: ';'.join(str(int(v)) for v in r if v > 0) or np.nan, axis=1)


def cat(s: pd.Series, codes: dict) -> pd.Series:
    """Map raw codes to canonical categories. `codes` lists every documented code; None marks
    a documented missing code. Negative codes (the BFS missing codes from 2000 on) become NaN
    silently; other codes not listed become NaN and are logged."""
    x = _to_float(s)
    unknown = x.notna() & (x >= 0) & ~x.isin(list(codes))
    if unknown.any():
        logging.warning(
            f"  {s.name}: {int(unknown.sum()):,} values with codes the codebook does not "
            f"list set to NaN: {x[unknown].value_counts().head(8).to_dict()}")
    return x.map(codes)


def flag(s: pd.Series, yes: Iterable[float] = (1,), no: Iterable[float] = (2,),
         na: Iterable[float] = ()) -> pd.Series:
    """1 for codes in `yes`, 0 for codes in `no`, NaN otherwise. Codes in none of `yes`, `no`,
    `na` and not negative are logged."""
    x = _to_float(s)
    yes, no, na = list(yes), list(no), list(na)
    unknown = x.notna() & (x >= 0) & ~x.isin(yes + no + na)
    if unknown.any():
        logging.warning(f"  {s.name}: {int(unknown.sum()):,} values that are neither yes nor no "
                        f"set to NaN: {x[unknown].value_counts().head(8).to_dict()}")
    return pd.Series(np.select([x.isin(yes), x.isin(no)], [1.0, 0.0], np.nan), index=s.index)


def any_flag(*flags: pd.Series) -> pd.Series:
    """1 if any flag is 1, 0 if all are 0, NaN otherwise."""
    df = pd.concat(flags, axis=1)
    return df.max(axis=1).where(df.notna().all(axis=1) | (df == 1).any(axis=1))


def bands(s: pd.Series, codes: dict) -> tuple[pd.Series, pd.Series]:
    """(lower, upper) bounds of banded answers; `codes` maps code → (lower, upper), with None
    for missing codes and None as upper bound of an open top band."""
    lo = cat(s, {k: (v[0] if v else None) for k, v in codes.items()})
    hi = _to_float(s).map({k: (v[1] if v else None) for k, v in codes.items()})
    return lo.astype(float), hi.astype(float)


def hhmm_to_min(s: pd.Series, na: Iterable[float] = ()) -> pd.Series:
    """Clock times written as HHMM (1994) → minutes after midnight."""
    x = num(s, na=na)
    h, m = x // 100, x % 100
    bad = (m >= 60) | (h > 36)
    if bad.any():
        logging.warning(f"  {s.name}: {int(bad.sum()):,} values are no HHMM time, set to NaN: "
                        f"{x[bad].value_counts().head(5).to_dict()}")
    return (h * 60 + m).mask(bad)


_DATE_FORMATS = ('%d.%m.%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%d%b%Y', '%Y%m%d')
_SPSS_EPOCH = pd.Timestamp('1582-10-14')


def parse_date(s: pd.Series) -> pd.Series:
    """Dates as 'YYYY-MM-DD'. Takes whichever of the common export formats parses most values
    (so day/month order is settled by the data); numbers are read as SPSS seconds."""
    if pd.api.types.is_numeric_dtype(s) and s.median() > 1e10:
        dates, fmt = _SPSS_EPOCH + pd.to_timedelta(s, unit='s'), 'SPSS seconds'
    else:
        txt = s.astype('string').str.strip().str.split(' ').str[0]
        dates, fmt = max(((pd.to_datetime(txt, format=f, errors='coerce'), f)
                          for f in _DATE_FORMATS), key=lambda t: t[0].notna().sum())
    if dates.notna().any():
        logging.info(f"  {s.name}: read as {fmt}, {dates.notna().mean():.1%} parsed, "
                     f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")
    return dates.dt.strftime('%Y-%m-%d')


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def standardize(codebook: ModuleType, raw: dict[str, pd.DataFrame],
                points: tuple[pd.DataFrame, pd.DataFrame] | None = None
                ) -> dict[str, pd.DataFrame]:
    """{table: canonical DataFrame indexed by its id} for one year. `points` (postcode and
    municipality points, `geo.load_points`) are needed when `codebook.XY_CRS` is None. Legs
    that repeat another leg exactly are dropped (logged)."""
    year = codebook.YEAR
    tables = {}
    for name in schema.TABLES:
        decode = getattr(codebook, name, None)
        if decode is None:
            logging.info(f"{year} {name}: not surveyed")
            tables[name] = pd.DataFrame(columns=['hhnr', 'person_no', 'journey_no'])
            continue
        logging.info(f"{year} {name}: decoding")
        tables[name] = decode(raw)
    tables['legs'] = _drop_duplicate_legs(tables['legs'])
    hh, members, people, vehicles, trips, legs = (tables[n] for n in (
        'households', 'household_members', 'people', 'vehicles', 'trips', 'legs'))
    journeys = {kind: tables[f'{kind}_journeys'] for kind in ('day', 'overnight')}
    places = tables['journey_places']

    _add_ids(year, hh, people, trips, legs)
    for kind, df in journeys.items():
        _add_journey_ids(year, kind, df, people)
    _add_place_ids(year, places, journeys)
    _add_member_ids(year, members, people, hh)
    _add_vehicle_ids(year, vehicles, members, hh)
    _link_legs_to_vehicles(legs, vehicles)
    for df in (people, members):
        for col in ('license_car', 'license_motorbike'):
            if col in df:
                # Not asked under 18, when nobody can hold one.
                df[col] = df[col].mask(df[col].isna() & (df['age'] < 18), 0)
    # Not asked under 15 from 2010; emptied in every year, so the years agree.
    child = people['age'] < 15
    blanked = {c: int((child & people[c].notna()).sum())
               for c in ('education', 'employment', 'employed_fulltime')}
    for col in blanked:
        people[col] = people[col].mask(child)
    logging.info(f"  Target persons under 15: {int(child.sum()):,}; values emptied: {blanked}")
    located = {'households': ('home',), 'trips': ('orig', 'dest'), 'legs': ('orig', 'dest'),
               'day_journeys': ('orig', 'dest'), 'overnight_journeys': ('orig', 'dest'),
               'journey_places': ('place',)}
    for name, prefixes in located.items():
        if len(tables[name]):
            crs = getattr(codebook, 'XY_CRS_BY_TABLE', {}).get(name, codebook.XY_CRS)
            for prefix in prefixes:
                _locate(tables[name], prefix, crs, points)
    _add_main_modes(trips, legs)
    _log_date_weekday(people)
    _log_trip_leg_consistency(trips, legs)
    _log_leg_distance_sources(legs)
    return {name: _conform(year, name, df) for name, df in tables.items()}


def _drop_duplicate_legs(legs: pd.DataFrame) -> pd.DataFrame:
    """Drop legs that repeat another leg in every decoded column, keys included (1994 has a
    few, all in one household)."""
    dup = legs.duplicated()
    if dup.any():
        trips = legs.loc[dup, ['hhnr', 'person_no', 'trip_no']].drop_duplicates()
        logging.warning(f"  Dropping {int(dup.sum()):,} legs that repeat another leg exactly, in "
                        f"{len(trips):,} trips (HHNR, person, trip: "
                        f"{trips.head(5).values.tolist()})")
    return legs[~dup]


def _id_part(s: pd.Series, what: str) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce')
    if x.isna().any():
        raise DataError(f"{int(x.isna().sum()):,} rows without a {what}")
    return x.astype('int64').astype(str)


def _add_ids(year: int, hh: pd.DataFrame, people: pd.DataFrame, trips: pd.DataFrame,
             legs: pd.DataFrame) -> None:
    """In place: hh / person / trip / leg ids. Leg numbers that restart in every trip (1994)
    are renumbered over the survey day."""
    hh['hh_id'] = f'{year}_' + _id_part(hh['hhnr'], 'household number')
    for df in (people, trips, legs):
        df['hh_id'] = f'{year}_' + _id_part(df['hhnr'], 'household number')
        df['person_id'] = df['hh_id'] + '_' + _id_part(df['person_no'], 'target person number')
    for df in (trips, legs):
        df['trip_id'] = df['person_id'] + '_' + _id_part(df['trip_no'], 'trip number')
    keyed = legs.assign(_n=pd.to_numeric(legs['leg_no'], errors='coerce'))
    if keyed.duplicated(['person_id', '_n']).any():
        if keyed.duplicated(['trip_id', '_n']).any():
            raise DataError(_repeated_legs_report(keyed))
        logging.info("  Leg numbers restart in every trip: renumbering over the survey day")
        legs['leg_no'] = (keyed.sort_values(['person_id', 'trip_no', '_n'])
                          .groupby('person_id').cumcount() + 1)
    legs['leg_id'] = legs['person_id'] + '_' + _id_part(legs['leg_no'], 'leg number')
    orphans = ~legs['trip_id'].isin(trips['trip_id'])
    if orphans.any():
        logging.warning(f"  {int(orphans.sum()):,} legs belong to no trip")


def _add_journey_ids(year: int, kind: str, journeys: pd.DataFrame, people: pd.DataFrame
                     ) -> None:
    """In place: journey ids, `<person_id>_<d|o><journey no.>`, and `weight_detailed`: the
    person weight × journeys reported / journeys detailed, as the BFS prescribes for the
    detailed journeys (up to 3 per person, picked at random). Logs how the journeys listed
    relate to the counts the people reported."""
    if journeys.empty:
        return
    _set_journey_ids(year, journeys, kind[0])
    per_person = journeys.groupby('person_id')
    n_listed = per_person['journey_id'].transform('size')
    n_detailed = per_person['is_detailed'].transform('sum')
    journeys['weight_detailed'] = (journeys['weight'] * n_listed / n_detailed).where(
        journeys['is_detailed'] == 1)
    module = people['extra_module'] == f'{kind}_journeys'
    reported = people.set_index('person_id')[f'n_{kind}_journeys']
    listed = per_person.size().reindex(reported.index, fill_value=0)
    has = reported > 0
    orphans = ~journeys['person_id'].isin(people['person_id'])
    logging.info(f"  {kind.capitalize()} journeys: module for {int(module.sum()):,} people "
                 f"({module.mean():.1%}), {int(has.sum()):,} reporting any; "
                 f"{len(journeys):,} listed, {int(journeys['is_detailed'].sum()):,} detailed; "
                 f"listed = reported for {(listed[has] == reported[has]).mean():.1%} of people; "
                 f"{int(orphans.sum()):,} journeys of no target person")


def _add_place_ids(year: int, places: pd.DataFrame, journeys: dict[str, pd.DataFrame]
                   ) -> None:
    """In place: journey and place ids (`<journey_id>_<place no.>`) of the journeys' stops
    and further overnight places, and the journey's `weight_detailed`."""
    if places.empty:
        return
    _set_journey_ids(year, places, places['place_type'].map(_PLACE_JOURNEY_TAG))
    places['place_id'] = places['journey_id'] + '_' + _id_part(places['place_no'], 'place number')
    weights = pd.concat([df.set_index('journey_id')['weight_detailed']
                         for df in journeys.values() if len(df)])
    places['weight_detailed'] = places['journey_id'].map(weights)
    per_type = places.groupby('place_type')['journey_id'].agg(['size', 'nunique'])
    orphans = ~places['journey_id'].isin(weights.index)
    undetailed = ~orphans & places['weight_detailed'].isna()
    logging.info(f"  Journey places: {per_type.to_dict('index')} (places, journeys); "
                 f"{int(orphans.sum()):,} of no journey, {int(undetailed.sum()):,} of journeys "
                 f"not detailed")


_PLACE_JOURNEY_TAG = {'stop': 'd', 'overnight': 'o'}


def _set_journey_ids(year: int, df: pd.DataFrame, tag: str | pd.Series) -> None:
    """In place: hh, person and journey ids, `<person_id>_<tag><journey no.>` with tag 'd'
    (day journey) or 'o' (overnight journey)."""
    df['hh_id'] = f'{year}_' + _id_part(df['hhnr'], 'household number')
    df['person_id'] = df['hh_id'] + '_' + _id_part(df['person_no'], 'target person number')
    df['journey_id'] = (df['person_id'] + '_' + tag
                        + _id_part(df['journey_no'], 'journey number'))


def _add_member_ids(year: int, members: pd.DataFrame, people: pd.DataFrame,
                    hh: pd.DataFrame) -> None:
    """In place: member ids and, where a member is a target person, their person id (from the
    data up to 2010, inferred from 2015: `_infer_target_members`)."""
    members['hh_id'] = f'{year}_' + _id_part(members['hhnr'], 'household number')
    members['member_id'] = (members['hh_id'] + '_m'
                            + _id_part(members['member_no'], 'member number'))
    if 'person_no' in members:
        no = pd.to_numeric(members['person_no'], errors='coerce')
        selected = (members['hh_id'] + '_' + no.astype('Int64').astype(str)).where(no > 0)
        interviewed = selected.isin(people['person_id'])
        members['person_id'] = selected.where(interviewed)
        logging.info(f"  Members selected as target person: {int(selected.notna().sum()):,}; "
                     f"{int((selected.notna() & ~interviewed).sum()):,} without an interview "
                     f"(no person id)")
    else:
        members['person_id'] = _infer_target_members(members, people)
    per_hh = members.groupby('hh_id').size().reindex(hh['hh_id'], fill_value=0).to_numpy()
    size = pd.to_numeric(hh['n_people'], errors='coerce').to_numpy()
    found = people['person_id'].isin(members['person_id'])
    logging.info(f"  Members: {len(members):,}; list = household size in "
                 f"{np.mean(per_hh == size):.1%} of households, size − 1 in "
                 f"{np.mean(per_hh == size - 1):.1%}; {found.mean():.1%} of target persons "
                 f"found among them")


def _infer_target_members(members: pd.DataFrame, people: pd.DataFrame) -> pd.Series:
    """Person ids for members, for years whose data does not say which member is the target
    person (2015 on: one target person per household). Member 1 if that member's age and sex
    match the target person in ≥ 95 % of households (the questionnaire asks for the other
    members after the respondent); otherwise the member with the target person's age and sex,
    where unique."""
    if people['hh_id'].duplicated().any():
        raise DataError("Inferring target members needs one target person per household")
    target = people.set_index('hh_id')[['person_id', 'age', 'sex']]
    j = members[['hh_id', 'member_no', 'age', 'sex']].join(target, on='hh_id', rsuffix='_tp')
    same = (j['age'] == j['age_tp']) & (j['sex'] == j['sex_tp'])
    first = pd.to_numeric(j['member_no'], errors='coerce') == 1
    share_first = same[first & j['person_id'].notna()].mean()
    if share_first >= 0.95:
        logging.info(f"  Target person = member 1 (age and sex agree in {share_first:.1%} of "
                     f"households)")
        return j['person_id'].where(first)
    unique = same & (same.groupby(j['hh_id']).transform('sum') == 1)
    logging.warning(f"  Member 1 is the target person in only {share_first:.1%} of households: "
                    f"matched by age and sex instead, unique in "
                    f"{unique.groupby(j['hh_id']).any().mean():.1%} of households")
    return j['person_id'].where(unique)


def _add_vehicle_ids(year: int, vehicles: pd.DataFrame, members: pd.DataFrame,
                     hh: pd.DataFrame) -> None:
    """In place: vehicles numbered per household and type (the "2nd car", as the leg question
    asks), vehicle ids and main drivers' member ids. Logs how the raw numbering relates."""
    if vehicles['vehicle_type'].isna().any():
        raise DataError(f"{int(vehicles['vehicle_type'].isna().sum()):,} vehicles of unknown type")
    vehicles['hh_id'] = f'{year}_' + _id_part(vehicles['hhnr'], 'household number')
    raw_no = pd.to_numeric(vehicles['vehicle_no'], errors='coerce')
    vehicles['vehicle_no'] = (vehicles.assign(_n=raw_no)
                              .sort_values(['hh_id', 'vehicle_type', '_n'])
                              .groupby(['hh_id', 'vehicle_type']).cumcount() + 1)
    vehicles['vehicle_id'] = (vehicles['hh_id'] + '_' + vehicles['vehicle_type']
                              + vehicles['vehicle_no'].astype(str))
    if vehicles['vehicle_id'].duplicated().any():
        raise DataError("Duplicated vehicles")
    car = vehicles['vehicle_type'] == 'car'
    kept = raw_no == vehicles['vehicle_no']
    n_rows = vehicles[car].groupby('hh_id').size().reindex(hh['hh_id'], fill_value=0)
    n_cars = pd.to_numeric(hh['n_cars'], errors='coerce').to_numpy()
    logging.info(f"  Vehicles: {int(car.sum()):,} cars, {int((~car).sum()):,} motorbikes; raw "
                 f"number = number within type for {kept[car].mean():.1%} of cars, "
                 f"{kept[~car].mean():.1%} of motorbikes; car rows = n_cars in "
                 f"{np.mean(n_rows.to_numpy() == n_cars):.1%} of households")
    if 'main_driver_nos' in vehicles:
        nos = vehicles['main_driver_nos'].dropna().str.split(';').explode()
        ids = vehicles.loc[nos.index, 'hh_id'] + '_m' + nos
        vehicles['main_driver_ids'] = ids.groupby(level=0).agg(';'.join)
        if len(ids):
            logging.info(f"  Main drivers: {ids.isin(members['member_id']).mean():.1%} found "
                         f"among the household members")


def _link_legs_to_vehicles(legs: pd.DataFrame, vehicles: pd.DataFrame) -> None:
    """In place: `vehicle_id` of the household car a car leg used. The car number was not
    asked in households with a single car; it is then 1."""
    if 'car_source' not in legs:
        return
    household = legs['car_source'] == 'household'
    car_no = pd.to_numeric(legs['car_no'], errors='coerce') if 'car_no' in legs \
        else pd.Series(np.nan, index=legs.index)
    n_cars = vehicles[vehicles['vehicle_type'] == 'car'].groupby('hh_id').size()
    car_no = car_no.mask(household & car_no.isna() & (legs['hh_id'].map(n_cars) == 1), 1)
    vehicle_id = (legs['hh_id'] + '_car' + car_no.astype('Int64').astype(str)
                  ).where(household & (car_no > 0))
    legs['vehicle_id'] = vehicle_id
    car = legs['mode'] == 'car'
    answered = legs.loc[car, 'car_source'].notna().groupby(legs.loc[car, 'is_passenger']).mean()
    logging.info(f"  Car legs with a car source: drivers {answered.get(0.0, np.nan):.1%}, "
                 f"passengers {answered.get(1.0, np.nan):.1%}")
    if household.any():
        found = vehicle_id[household].isin(vehicles['vehicle_id'])
        logging.info(f"  Car legs in a household car: {int(household.sum()):,}; "
                     f"{found.mean():.1%} linked to a vehicle")


def _repeated_legs_report(keyed: pd.DataFrame) -> str:
    """Error text for leg numbers that repeat within a trip: how many are missing numbers, how
    many are exact duplicate rows, and examples."""
    dup = keyed[keyed.duplicated(['trip_id', '_n'], keep=False)]
    show = [c for c in ('trip_id', 'leg_no', 'dep_min', 'arr_min', 'duration_min', 'dist_km',
                        'mode', 'purpose') if c in dup]
    return (f"Leg numbers repeat within a trip: {len(dup):,} legs in "
            f"{dup['trip_id'].nunique():,} trips; {int(keyed['_n'].isna().sum()):,} legs have no "
            f"leg number, {int(keyed.duplicated(keep=False).sum()):,} legs are exact duplicates "
            f"of another leg. Examples:\n"
            f"{dup.sort_values(['trip_id', '_n'])[show].head(12).to_string()}")


def _locate(df: pd.DataFrame, prefix: str, crs: str | None,
            points: tuple[pd.DataFrame, pd.DataFrame] | None) -> None:
    if crs is not None:
        if f'{prefix}_x' in df:
            df[f'{prefix}_x'], df[f'{prefix}_y'] = geo.to_lv95(
                df[f'{prefix}_x'], df[f'{prefix}_y'], crs)
        return
    if points is None:
        raise DataError("No coordinates in this year: postcode points needed")
    geo.locate_from_points(df, prefix, points)
    in_ch = f'{prefix}_in_switzerland'
    if in_ch not in df and prefix != 'home':
        # Only Swiss postcodes and municipalities are in the point lists.
        df[in_ch] = df[f'{prefix}_x'].notna().astype(float).where(df[f'{prefix}_x'].notna())


def _add_main_modes(trips: pd.DataFrame, legs: pd.DataFrame) -> None:
    """In place: `mode_main_ranked` (highest leg mode in `schema.MODES`) and
    `mode_main_longest` (mode of the longest leg, ties by rank). Empty for trips with a leg of
    unknown mode (e.g. pseudo legs) resp. unknown distance, as the BFS does."""
    rank = {m: i for i, m in enumerate(schema.MODES)}
    lg = legs[['trip_id', 'mode', 'dist_km']].assign(rank=legs['mode'].map(rank))
    no_mode = lg.loc[lg['mode'].isna(), 'trip_id'].unique()
    no_dist = lg.loc[lg['dist_km'].isna(), 'trip_id'].unique()
    ranked = (lg.dropna(subset=['mode']).sort_values('rank')
              .drop_duplicates('trip_id').set_index('trip_id')['mode'])
    longest = (lg.dropna(subset=['mode', 'dist_km'])
               .sort_values(['dist_km', 'rank'], ascending=[False, True])
               .drop_duplicates('trip_id').set_index('trip_id')['mode'])
    trips['mode_main_ranked'] = trips['trip_id'].map(ranked).mask(trips['trip_id'].isin(no_mode))
    trips['mode_main_longest'] = (trips['trip_id'].map(longest)
                                  .mask(trips['trip_id'].isin(np.union1d(no_mode, no_dist))))
    both = trips['mode_main_ranked'].notna() & trips['mode_main_longest'].notna()
    if both.any():
        same = (trips.loc[both, 'mode_main_ranked'] == trips.loc[both, 'mode_main_longest']).mean()
        logging.info(f"  Main mode: ranked and longest-leg agree on {same:.1%} of trips")


def _log_date_weekday(people: pd.DataFrame) -> None:
    """Log how often `survey_date` falls on the reported `weekday` — checks the date format
    picked by `parse_date`."""
    if 'survey_date' not in people:
        return
    dates = pd.to_datetime(people['survey_date'], errors='coerce')
    known = dates.notna() & people['weekday'].notna()
    if known.any():
        agree = (dates[known].dt.dayofweek + 1 == people.loc[known, 'weekday']).mean()
        logging.info(f"  Survey date falls on the reported weekday for {agree:.1%} of people")


def _log_trip_leg_consistency(trips: pd.DataFrame, legs: pd.DataFrame) -> None:
    """Log how trip totals relate to their legs — checks the codebook's reading of the trip
    duration variables (duration incl. waits ≥ in-motion time = Σ leg durations)."""
    per_trip = legs.groupby('trip_id').agg(n=('leg_id', 'size'),
                                           dur=('duration_min', 'sum'),
                                           dist=('dist_km', 'sum'))
    t = trips.set_index('trip_id').join(per_trip, how='inner')
    checks = (
        ('n_legs = legs listed', t['n_legs'] == t['n'], ['n_legs']),
        ('in_motion_min = Σ leg durations (±1)',
         (t['in_motion_min'] - t['dur']).abs() <= 1, ['in_motion_min']),
        ('duration_min ≥ in_motion_min',
         t['duration_min'] >= t['in_motion_min'] - 1, ['duration_min', 'in_motion_min']),
        ('dist_km = Σ leg dist_km (±0.1)', (t['dist_km'] - t['dist']).abs() <= 0.1, ['dist_km']),
    )
    for name, ok, needed in checks:
        known = t[needed].notna().all(axis=1)
        if known.any():
            logging.info(f"  Trips vs legs: {name}: {ok[known].mean():.1%} of "
                         f"{int(known.sum()):,} trips")


_MODE_GROUPS = {
    'walk': ('walk',),
    'bike': ('rbike', 'anybike', 'ebike25', 'ebike45'),
    'car': ('car',),
    'transit': ('transit_rail', 'transit_bus', 'transit_tram'),
}


def _log_leg_distance_sources(legs: pd.DataFrame) -> None:
    """Log, per mode group, how a leg's `dist_km` relates to the respondent's estimate: equal
    (the estimate is used), a near-constant ratio (a corrected estimate) or neither (routed).
    The documentation does not say for every year."""
    est = legs['dist_estimated_km']
    both = legs['dist_km'].notna() & (est > 0)
    out = {}
    for group, modes in _MODE_GROUPS.items():
        m = both & legs['mode'].isin(modes)
        if m.sum() >= 100:
            equal = ((legs.loc[m, 'dist_km'] - est[m]).abs() <= 0.01).mean()
            ratio = (legs.loc[m, 'dist_km'] / est[m]).quantile([0.1, 0.5, 0.9]).round(2)
            out[group] = f"{equal:.0%} equal, ratio p10/50/90 {ratio.tolist()}, n {int(m.sum()):,}"
    if out:
        logging.info(f"  Leg dist_km vs the respondent's estimate: {out}")


_DISTANCES = ('dist_km', 'dist_estimated_km', 'dist_domestic_km')
_DIST_DECIMALS = 3                              # km → 1 m
_FRACTIONAL = ('weight', 'weight_detailed', 'duration_h', *_DISTANCES)  # floats even if whole


def _conform(year: int, name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The schema's columns in order (missing ones empty), vocabularies checked, distances
    rounded, whole-number columns as nullable integers, indexed by the table's id."""
    cols = list(schema.TABLES[name])
    out = (df.assign(year=year).reindex(columns=cols)
           .round({c: _DIST_DECIMALS for c in _DISTANCES}))
    for col in cols:
        # Codes, flags, counts, times: integers, with missing values left empty (not 8400.0).
        s = out[col]
        if (col not in _FRACTIONAL and pd.api.types.is_float_dtype(s) and s.notna().any()
                and (s.dropna() % 1 == 0).all()):
            out[col] = s.astype('Int64')
    for col in cols:
        vocab = schema.CATEGORIES.get(f'{name}.{col}', schema.CATEGORIES.get(col))
        if vocab is not None:
            bad = out[col].notna() & ~out[col].isin(vocab)
            if bad.any():
                raise DataError(f"{year} {name}.{col}: values outside the schema vocabulary "
                                f"{sorted(set(out.loc[bad, col]))}; fix codebook_{year}.py")
    if out[cols[0]].duplicated().any():
        raise DataError(f"{year} {name}: duplicated {cols[0]}, e.g. "
                        f"{out.loc[out[cols[0]].duplicated(), cols[0]].head(3).tolist()}")
    filled = out.notna().mean().mul(100)
    logging.info(f"  {name}: {len(out):,} rows; empty: {[c for c in cols if filled[c] == 0]}; "
                 f"partly filled (%): "
                 f"{ {c: round(filled[c]) for c in cols if 0 < filled[c] < 100} }")
    return out.set_index(cols[0])
