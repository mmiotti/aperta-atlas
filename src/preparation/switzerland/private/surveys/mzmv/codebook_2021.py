"""
MTMC 2021 (Mikrozensus Mobilität und Verkehr 2021) → canonical schema.

Source documentation: `2023-06-05-Datenbankstruktur_MZMV2021_mit_Geodaten.xlsx`.

Year specifics:
  - One target person per household → person no. 1.
  - Leg modes renumbered (e-bikes 3/4, Postauto merged into bus, ride-hailing added).
  - Education recoded to 9 levels (f40120), with universities of applied sciences and
    universities merged.
  - Full-time employment is only available as the employment rate (f40920, %); full-time
    means ≥ 90 %, as in BFS usage.
  - The BFS trip main mode now comes in two versions (by ranking / by distance); the
    standardized tables recompute both from the legs for every year.
  - Journeys: companions no longer asked; overnight journeys are those that ended (not
    started) in the 4 months before the interview. Distances as in 2015 (imputed version).
  - Vehicles: car engine size no longer asked (motorbikes only). Registry data adds fuel and
    electricity consumption (NEDC). Cars that by the registry use both electricity and fuel
    (electricity consumption above zero, and fuel consumption or CO2 above zero) are classed
    as plug-in hybrids, whatever fuel was reported; so are hybrids with an electricity
    consumption but no fuel data, and range extenders. Unlike 2015, a CO2 cut would sit too
    close to newer ordinary hybrids (Prius from 2016: 70 g/km NEDC). Empty CO2 and fuel
    consumption of electric cars found in the registry are set to 0, as in 2015.
"""

import logging
import textwrap

import pandas as pd

from preparation.switzerland.private.surveys.mzmv import codebook_2005, codebook_2010, codebook_2015
from preparation.switzerland.private.surveys.mzmv.codebook_2010 import endpoints
from preparation.switzerland.private.surveys.mzmv.engine import (
    any_flag, cat, flag, num, numbers, parse_date)

YEAR = 2021
RAW_DIR = 'MZMV2021_mit_Geo/4_DB_csv'
RAW_FILES = {'hh': 'haushalte', 'hp': 'haushaltspersonen', 'zp': 'zielpersonen',
             'fz': 'fahrzeuge', 'wg': 'wege', 'et': 'etappen',
             'dj': 'tagesreisen', 'oj': 'reisenmueb'}
READ_KWS = dict(sep=';', encoding='latin-1')
XY_CRS = 'EPSG:4326'

SEX = {1: 'male', 2: 'female'}
PURPOSE = codebook_2005.PURPOSE
LEISURE = codebook_2010.LEISURE
EMPLOYMENT = codebook_2010.EMPLOYMENT
WEATHER = codebook_2010.WEATHER
VEHICLE_TYPE = codebook_2005.VEHICLE_TYPE
CAR_SOURCE = codebook_2005.CAR_SOURCE
FUEL = codebook_2015.FUEL

households = codebook_2015.households
household_members = codebook_2015.household_members
trips = codebook_2015.trips

# f51300: Benutztes Verkehrsmittel
MODE = {
    1: 'walk',                  # Zu Fuss
    2: 'rbike',                 # Velo
    3: 'ebike25',               # E-Bike ohne Kontrollschild
    4: 'ebike45',               # E-Bike mit gelbem Kontrollschild
    5: 'motorbike_small',       # Mofa, Motorfahrrad
    6: 'motorbike_small',       # Kleinmotorrad (ab 16 J.)
    7: 'motorbike',             # Motorrad als Fahrer
    8: 'motorbike',             # Motorrad als Mitfahrer
    9: 'car',                   # Auto als Fahrer
    10: 'car',                  # Auto als Mitfahrer
    11: 'transit_rail',         # Bahn/Zug
    12: 'transit_bus',          # Bus, Postauto, Schulbus
    13: 'transit_tram',         # Tram/Metro
    14: 'taxi',                 # Taxi (im "klassischen" Sinn)
    15: 'taxi',                 # Taxi-ähnliche Fahrdienste (z.B. Uber)
    16: 'coach',                # Reisecar
    17: 'truck',                # Lastwagen
    18: 'boat',                 # Schiff, Boot
    19: 'plane',                # Flugzeug/Luftfahrzeug
    20: 'aerialway',            # Zahnradbahn, Standseilbahn, Seilbahn, Sessellift, Skilift
    21: 'micro',                # Fahrzeugähnliche Geräte
    95: 'other',                # Anderes
}
PASSENGER, DRIVER = (8, 10), (7, 9)

# f40120: Höchste Ausbildung
EDUCATION = {
    1: 'none',                          # Keine (oder obligatorische Schule nicht abgeschlossen)
    2: 'compulsory',                    # Obligatorische Schule
    3: 'upper_secondary_general',       # Allgemeinbildung ohne Maturität (DMS, FMS)
    4: 'upper_secondary_vocational',    # Berufliche Grundbildung, Berufslehre
    5: 'upper_secondary_general',       # Maturität (gymnasial, Berufs-, Fach-), Lehrerseminar
    6: 'tertiary_professional',         # Höhere Berufsbildung (eidg. Fachausweis / Diplom)
    7: 'tertiary_academic',             # Höhere Fachschule (HTL, HWV, HFG, HFS, TS)
    8: 'tertiary_academic',             # Fachhochschule oder Universität
    9: 'tertiary_academic',             # Doktorat, Habilitation
}

_FULLTIME_PCT = 90


def people(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zp = raw['zp']
    passes = {c: flag(zp[f'f41600_01{c}']) for c in 'abcdefg'}
    # In pass_season, not asked (-99; e.g. seven25 / Gleis 7 only up to age 24) means not held.
    season = {c: flag(zp[f'f41600_01{c}'], no=(2, -99)) for c in 'cdefg'}
    employment_pct = num(zp['f40920'])
    return pd.DataFrame({
        'hhnr': zp['hhnr'],
        'person_no': 1,
        'weight': num(zp['wp']),
        'age': num(zp['alter']),
        'sex': cat(zp['gesl'], SEX),
        'survey_date': parse_date(zp['vstag']),
        'weekday': num(zp['tag']),
        'is_mobile': flag(zp['f50200']),
        'license_car': flag(zp['f20400a']),
        'license_motorbike': flag(zp['f20400b']),
        'pass_ga': passes['a'],                                     # Generalabo
        'pass_halbtax': passes['b'],                                # Halbtaxabo
        'pass_regional': passes['c'],                               # Verbund-Abo
        # Verbund, Strecke, seven25 / Gleis 7, Junior-Karte, anderes
        'pass_season': any_flag(*season.values()),
        'education': cat(zp['f40120'], EDUCATION),
        'employment': cat(zp['estatus'], EMPLOYMENT),
        'employed_fulltime': (employment_pct >= _FULLTIME_PCT).astype(float)
                             .where(employment_pct.notna()),
        'weather': cat(zp['f50100a'], WEATHER),
        **codebook_2010.journey_counts(zp),
    })


def vehicles(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fz = raw['fz']
    engine_cc = num(fz['f31400'])                   # motorbikes only
    km = num(fz['f30900_31700'])
    reg = {
        **codebook_2015.registry(fz),
        'fuel_l_100km': num(fz['et_verbrauch']),
        'electricity_kwh_100km': num(fz['el_verbrauch']),
    }
    vehicle_type = cat(fz['fahrzeugart'], VEHICLE_TYPE)
    car = vehicle_type == 'car'
    fuel = _plugins_by_electricity(cat(fz['f30400'], FUEL), reg['electricity_kwh_100km'],
                                   reg['fuel_l_100km'], reg['co2_g_km'], car)
    return pd.DataFrame({
        'hhnr': fz['hhnr'],
        'weight': num(fz['wm']),
        'vehicle_type': vehicle_type,
        'vehicle_no': fz['fznum'],
        'registration_year': num(fz['f30600_31500']),
        'registration_month': num(fz['f30601_31501']),
        'odometer_km': num(fz['f30800_31600']),
        'km_per_year': km,                          # last 12 months
        'km_per_year_abroad': num(fz['f31000_31800']).mask(km == 0, 0),
        'engine_cc_min': engine_cc,
        'engine_cc_max': engine_cc,
        'fuel': fuel,
        'main_driver_nos': numbers(*(fz[f'f30700_hpnr{i}'] for i in range(1, 6))),
        **codebook_2015.zero_for_electric(reg, fuel, ['co2_g_km', 'fuel_l_100km']),
    })


def _plugins_by_electricity(fuel: pd.Series, electricity: pd.Series, fuel_l: pd.Series,
                            co2: pd.Series, car: pd.Series) -> pd.Series:
    """Cars that use both electricity and fuel by the registry, and hybrids with an
    electricity consumption, become plug-in hybrids, whatever fuel was reported."""
    charges = electricity > 0
    burns = (fuel_l > 0) | (co2 > 0)
    plugin = car & charges & (burns | (fuel == 'hybrid')) & (fuel != 'plugin_hybrid')
    table = pd.DataFrame({
        'cars': True,
        'electricity > 0': charges,
        'and fuel or CO2 > 0': charges & burns,
        '→ plug-in': plugin,
        'CO2 = 0': co2 == 0,
        'CO2 0-70, no electricity': (co2 > 0) & (co2 < 70) & ~charges,
        'no CO2': co2.isna(),
    }, index=fuel.index)[car].groupby(fuel[car].fillna('(no answer)')).sum()
    logging.info(f"  Cars by reported fuel and registry consumption; {int(plugin.sum()):,} → "
                 f"plug-in hybrid:\n"
                 + textwrap.indent(table.rename_axis(index=None).to_string(), ' ' * 6))
    return fuel.mask(plugin, 'plugin_hybrid')


def legs(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    et = raw['et']
    return pd.DataFrame({
        'hhnr': et['hhnr'],
        'person_no': 1,
        'trip_no': et['wegnr'],
        'leg_no': et['etnr'],
        'weight': num(et['wp']),
        'dep_min': num(et['f51100']),
        'arr_min': num(et['f51400']),
        'duration_min': num(et['e_dauer']),
        'dist_km': num(et['rdist']),
        'dist_estimated_km': num(et['schaetzdist']),
        'mode': cat(et['f51300'], MODE),
        'is_passenger': flag(et['f51300'], yes=PASSENGER, no=DRIVER, na=MODE),
        'car_occupancy': num(et['f51320']),
        'car_source': cat(et['f51310a'], CAR_SOURCE),
        'car_no': num(et['f51310b']),
        'purpose': cat(et['f52900'], PURPOSE),
        'leisure_activity': cat(et['f51700'], LEISURE),
        'is_abroad': flag(et['e_ausland']),
        'is_pseudo': flag(et['pseudo'], yes=(2,), no=(1,)),
        **endpoints(et),
    })


def day_journeys(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return codebook_2010.journeys(raw['dj'], 'day', MODE, PASSENGER, DRIVER, dist_version='b')


def overnight_journeys(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return codebook_2010.journeys(raw['oj'], 'overnight', MODE, PASSENGER, DRIVER,
                                  dist_version='b')


journey_places = codebook_2010.journey_places
