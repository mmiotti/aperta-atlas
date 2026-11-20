"""
MTMC 2015 (Mikrozensus Mobilität und Verkehr 2015) → canonical schema.

Source documentation: `2017-11-23-Datenbankstruktur_MZMV2015_mit_Geodaten_mit_Korrekturen.xlsx`,
`Herleitung der Wege_und_Ausgaenge_inklusiv_Anpassungen_beim_MZMV2015.pdf`.

Year specifics:
  - One target person per household (no ZIELPNR) → person no. 1.
  - E-bikes are their own leg modes (codes 20, 21) and household counts (f32200b/c); 'Velo'
    is now a bike without assist.
  - Trips: a change of leisure activity (f51700) now also starts a new trip.
  - The trip table has no summed distance estimate (only legs have `schaetzdist`, and not for
    every leg).
  - Household members: the data does not say which member is the target person (inferred by
    the engine). Up to five main drivers per car.
  - Fuel: the survey has no plug-in hybrid answer; plug-ins are mostly under 'Hybrid', but
    owners also report them as petrol or electric. Cars with a registry CO2 value above 0 and
    below 70 g/km (NEDC) are classed as plug-in, whatever fuel was reported: ordinary hybrids
    and combustion cars in the 2015 fleet are ≥ ~75 g/km, plug-ins mostly 30-60. Range
    extenders count as plug-ins too. The log shows the low end of the CO2 distribution by
    reported fuel to check the cut.
  - Registry data adds empty weight, payload, power, driven wheels, CO2 (labelled g/100 km
    in the codebook; the unit is checked from the values) and type-approval number. The
    registry has no CO2 value for battery-electric cars: 0 for electric cars found in it.
  - Journeys: distances in two versions; the second (f61600b etc.), which the BFS imputed
    where the respondent did not know and used for its main report, is taken.
"""

import logging
import textwrap

import numpy as np
import pandas as pd

from preparation.switzerland.private.surveys.mzmv import codebook_2005, codebook_2010
from preparation.switzerland.private.surveys.mzmv.codebook_2010 import endpoints
from preparation.switzerland.private.surveys.mzmv.engine import (
    any_flag, bands, cat, coord, flag, num, numbers, parse_date, text)

YEAR = 2015
RAW_DIR = 'MZMV2015_mit_Geo/4_DB_csv'
RAW_FILES = {'hh': 'haushalte', 'hp': 'haushaltspersonen', 'zp': 'zielpersonen',
             'fz': 'fahrzeuge', 'wg': 'wege', 'et': 'etappen',
             'dj': 'tagesreisen', 'oj': 'reisenmueb'}
READ_KWS = dict(sep=',', encoding='latin-1')
XY_CRS = 'EPSG:4326'

SEX = {1: 'male', 2: 'female'}
PURPOSE = codebook_2005.PURPOSE
DIRECTION = codebook_2005.DIRECTION
INCOME = codebook_2005.INCOME
LEISURE = codebook_2010.LEISURE
EMPLOYMENT = codebook_2010.EMPLOYMENT
WEATHER = codebook_2010.WEATHER
VEHICLE_TYPE = codebook_2005.VEHICLE_TYPE
CAR_SOURCE = codebook_2005.CAR_SOURCE
TRANSMISSION = codebook_2010.TRANSMISSION

# f30400: Treibstoff Auto (kept in 2021)
FUEL = {
    1: 'petrol',                # Benzin
    2: 'diesel',                # Diesel
    3: 'ethanol',               # E85 / Bioethanol
    4: 'gas',                   # Erdgas / Biogas (CNG), incl. bivalent with petrol
    5: 'gas',                   # Flüssiggas (LPG), incl. bivalent with petrol
    6: 'hybrid',                # Hybrid (Benzin/elektrisch und Diesel/elektrisch)
    7: 'electric',              # rein elektrisch
    8: 'plugin_hybrid',         # elektrisch mit Range-Extender (Reichweitenverlängerer)
    95: 'other',                # andere
}

# mf_antri (registry): Antriebsart (kept in 2021)
DRIVE = {
    1: 'front',                 # V. Vorderrad
    2: 'rear',                  # H. Hinterrad
    3: 'all_wheel',             # A. Allrad
    4: 'tracks',                # R. Raupen
    5: 'off_road',              # G. Geländegängig
}

_PLUGIN_CO2_MAX = 70            # g/km NEDC, see the module docstring

# f51300: Benutztes Verkehrsmittel
MODE = {
    1: 'walk',                  # Zu Fuss
    2: 'rbike',                 # Velo
    3: 'motorbike_small',       # Mofa, Motorfahrrad
    4: 'motorbike_small',       # Kleinmotorrad (ab 16 J.)
    5: 'motorbike',             # Motorrad als Fahrer
    6: 'motorbike',             # Motorrad als Mitfahrer
    7: 'car',                   # Auto als Fahrer
    8: 'car',                   # Auto als Mitfahrer
    9: 'transit_rail',          # Bahn/Zug
    10: 'transit_bus',          # Postauto
    11: 'transit_bus',          # Bus/Schulbus
    12: 'transit_tram',         # Tram/Metro
    13: 'taxi',                 # Taxi
    14: 'coach',                # Reisecar
    15: 'truck',                # Lastwagen
    16: 'boat',                 # Schiff, Boot
    17: 'plane',                # Flugzeug/Luftfahrzeug
    18: 'aerialway',            # Zahnradbahn, Standseilbahn, Seilbahn, Sessellift, Skilift
    19: 'micro',                # Fahrzeugähnliche Geräte
    20: 'ebike25',              # E-Bike ohne Kontrollschild
    21: 'ebike45',              # E-Bike mit gelbem Kontrollschild
    95: 'other',                # Anderes
}
PASSENGER, DRIVER = (6, 8), (5, 7)

# hhtyp: Typ des Privathaushalts (kept in 2021)
HH_TYPE = {
    10: 'single',               # Einpersonenhaushalt
    30: 'other',                # Nichtfamilienhaushalte
    210: 'couple_no_kids',      # Paare ohne Kinder
    220: 'couple_kids',         # Paare mit Kind(ern)
    230: 'single_parent',       # Einelternhaushalte mit Kind(ern)
}

# HAUSB: Höchste abgeschlossene Ausbildung (2010 codes up to 12, then split further)
EDUCATION = {
    **{k: v for k, v in codebook_2010.EDUCATION.items() if k <= 12},
    13: 'tertiary_professional',        # Höhere Berufsbildung mit eidg. Fachausweis
    14: 'tertiary_professional',        # Höhere Berufsbildung mit eidg. Diplom / Meisterdiplom
    15: 'tertiary_academic',            # Höhere Fachschule (HF) Technik / Wirtschaft
    16: 'tertiary_academic',            # Höhere Fachschule HWV, HFG, HFS, HTL
    17: 'tertiary_academic',            # Fachhochschule
    18: 'tertiary_academic',            # Pädagogische Hochschule
    19: 'tertiary_academic',            # Universität, ETH
}


def households(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hh = raw['hh']
    income_min, income_max = bands(hh['f20601'], INCOME)
    bikes = num(hh['f32200a'])
    ebikes_slow, ebikes_fast = num(hh['f32200b']), num(hh['f32200c'])
    return pd.DataFrame({
        'hhnr': hh['hhnr'],
        'weight': num(hh['wm']),
        'home_x': coord(hh['w_x']),
        'home_y': coord(hh['w_y']),
        'home_geocode_level': cat(hh['w_qal'], codebook_2010.GEOCODE_LEVEL),
        'home_postcode': num(hh['w_plz']),
        'home_municipality': num(hh['w_bfs']),
        'home_canton_no': num(hh['w_kanton']),
        'n_people': num(hh['hhgr']),
        'hh_type': cat(hh['hhtyp'], HH_TYPE),
        'income_min_chf': income_min,
        'income_max_chf': income_max,
        'n_cars': num(hh['f30100']),
        'n_car_parking': num(hh['f31100']),
        'n_motorbikes': num(hh['f31200']),
        'n_small_motorbikes': num(hh['f31900']),
        'n_mopeds': num(hh['f32100']),
        'n_bikes_any': bikes + ebikes_slow + ebikes_fast,
        'n_ebikes_slow': ebikes_slow,
        'n_ebikes_fast': ebikes_fast,
        'has_second_home': flag(hh['f10700']),
    })


def people(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zp = raw['zp']
    passes = {c: flag(zp[f'f41610{c}']) for c in 'abcdefg'}
    # In pass_season, not asked (-99; e.g. Gleis 7 only up to age 25) means not held.
    season = {c: flag(zp[f'f41610{c}'], no=(2, -99)) for c in 'cdefg'}
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
        'pass_regional': passes['c'],                               # Verbund-Abo, Inter-Abo
        # Verbund, Strecke, Gleis 7, Juniorkarte, anderes
        'pass_season': any_flag(*season.values()),
        'education': cat(zp['hausb'], EDUCATION),
        'employment': cat(zp['estatus'], EMPLOYMENT),
        'employed_fulltime': flag(zp['f40900'], yes=(1,), no=(2, 3)),
        'weather': cat(zp['f50100a'], WEATHER),
        **codebook_2010.journey_counts(zp),
    })


def household_members(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """No target-person number in the data (kept in 2021); the engine infers it."""
    hp = raw['hp']
    return pd.DataFrame({
        'hhnr': hp['hhnr'],
        'member_no': hp['hpnr'],
        'weight': num(hp['wm']),
        'age': num(hp['alter']),
        'sex': cat(hp['gesl'], SEX),
        'license_car': flag(hp['f20400a']),
        'license_motorbike': flag(hp['f20400b']),
    })


def vehicles(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fz = raw['fz']
    engine_cc = num(fz['f30500_31400'])             # -99 = purely electric
    km = num(fz['f30900_31700'])
    reg = registry(fz)
    vehicle_type = cat(fz['fahrzeugart'], VEHICLE_TYPE)
    car = vehicle_type == 'car'
    fuel = plugins_by_co2(cat(fz['f30400'], FUEL), reg['co2_g_km'], car)
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
        **zero_for_electric(reg, fuel, ['co2_g_km']),
    })


def registry(fz: pd.DataFrame) -> dict[str, pd.Series]:
    """Vehicle-registry (MOFIS) fields from 2015 on; 2021 adds consumption."""
    return {
        **codebook_2010.registry(fz),
        'weight_empty_kg': num(fz['leergew']),
        'payload_kg': num(fz['mf_nuzla']),
        'power_kw': num(fz['mf_leikw']),
        'drive': cat(fz['mf_antri'], DRIVE),
        'co2_g_km': co2_g_km(fz['mf_co2wt']),
        'type_approval_no': text(fz['mf_tysnr']),
    }


def co2_g_km(s: pd.Series) -> pd.Series:
    """Registry CO2 in g/km. The 2015 codebook labels it g/100 km; values of a few hundred
    are g/km, values in the thousands g/100 km."""
    co2 = num(s)
    median = co2.median()
    if median > 1000:
        logging.info(f"  {s.name}: median {median:.0f} → read as g/100 km, divided by 100")
        return co2 / 100
    logging.info(f"  {s.name}: median {median:.0f} → read as g/km")
    return co2


def zero_for_electric(reg: dict[str, pd.Series], fuel: pd.Series,
                      columns: list[str]) -> dict[str, pd.Series]:
    """The registry leaves CO2 (and fuel consumption) empty for battery-electric cars: 0 for
    electric cars found in the registry."""
    electric = (fuel == 'electric') & (reg['in_registry'] == 1)
    filled = {c: electric & reg[c].isna() for c in columns}
    logging.info(f"  Electric cars in the registry: {int(electric.sum()):,}; empty values set to "
                 f"0: {', '.join(f'{c} {int(f.sum()):,}' for c, f in filled.items())}")
    return {**reg, **{c: reg[c].mask(f, 0) for c, f in filled.items()}}


def plugins_by_co2(fuel: pd.Series, co2: pd.Series, car: pd.Series) -> pd.Series:
    """Cars with a registry CO2 value above 0 and below `_PLUGIN_CO2_MAX` become plug-in
    hybrids, whatever fuel was reported. Motorbikes are left alone (small ones are below the
    cut)."""
    plugin = car & (co2 > 0) & (co2 < _PLUGIN_CO2_MAX) & (fuel != 'plugin_hybrid')
    labels = ['no CO2', '0', '<30', '30-50', f'50-{_PLUGIN_CO2_MAX}', f'{_PLUGIN_CO2_MAX}-90',
              '90+']
    bins = pd.cut(co2[car], [0, 1, 30, 50, _PLUGIN_CO2_MAX, 90, np.inf], right=False,
                  labels=labels[1:]).astype(object).fillna('no CO2')
    table = (pd.crosstab(fuel[car].fillna('(no answer)'), bins)
             .reindex(columns=labels, fill_value=0).rename_axis(index=None, columns=None)
             .to_string())
    logging.info(f"  Cars by reported fuel and registry CO2 (g/km); {int(plugin.sum()):,} above "
                 f"0 and below {_PLUGIN_CO2_MAX} → plug-in hybrid:\n"
                 + textwrap.indent(table, ' ' * 6))
    return fuel.mask(plugin, 'plugin_hybrid')


def trips(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wg = raw['wg']
    return pd.DataFrame({
        'hhnr': wg['hhnr'],
        'person_no': 1,
        'trip_no': wg['wegnr'],
        'weight': num(wg['wp']),
        'dep_min': num(wg['f51100']),
        'arr_min': num(wg['f51400']),
        'duration_min': num(wg['dauer2']),
        'in_motion_min': num(wg['dauer1']),
        'dist_km': num(wg['w_rdist']),
        'n_legs': num(wg['w_etappen']),
        'has_pseudo_leg': flag(wg['pseudo'], yes=(2,), no=(1,)),
        'purpose': cat(wg['wzweck1'], PURPOSE),
        'direction': cat(wg['wzweck2'], DIRECTION),
        'leisure_activity': cat(wg['f51700_weg'], LEISURE),
        **endpoints(wg),
    })


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
