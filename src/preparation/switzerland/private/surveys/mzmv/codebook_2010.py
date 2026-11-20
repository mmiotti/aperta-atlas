"""
MTMC 2010 (Mikrozensus Mobilität und Verkehr 2010) → canonical schema.

Source documentation: `2014-01-31-Datenbankstruktur_MZMV2010_mit_Geodaten.xlsx`,
`EA-2011-12-22_Datenbankbeschreibung.pdf`, `EA-2011-12-22_Glossar.pdf`.

Year specifics:
  - Up to two target persons per household (ZIELPNR), as before.
  - First year with routing: `rdist` is routed for motorised legs and the respondent's
    estimate for walk/bike legs; WGS84 coordinates (CH1903 also delivered); route geometries.
  - Trips: a trip continues while the purpose stays the same or is 'transfer', and the gap is
    ≤ 1 h (2005: 'transfer' always ended a trip).
  - Bikes: e-bikes are not told apart at leg level → 'anybike'. Household e-bike counts exist
    only for the module-2 subsample and are not used.
  - Codes that differ from 2015 although the variable names match: transit-pass letters
    (a = half-fare, b/c = GA 1st/2nd class), HAUSB from code 13 on, hhtyp (single = 100).
  - Employment from here on is the BFS labour-force status (ESTATUS), not self-assessed.
  - Vehicles: one number across cars and motorbikes (fznum; renumbered per type by the
    engine). Fuel merges hybrid, E85 and gas ('alternative'). Up to two main drivers per car.
    First year with vehicle-registry (MOFIS) data for matched vehicles: gross weight,
    cylinders, gearbox, energy label, emission code.
  - Journeys (modules 1a / 1b): the questions and variable names hold to 2021 (`journeys`).
    Purposes now split sport; the main mode is the one of the way out, with its own return
    mode; distances also in Switzerland (asked for journeys abroad only).
"""

import logging

import pandas as pd

from preparation.switzerland.private.surveys.mzmv import codebook_2005
from preparation.switzerland.private.surveys.mzmv.engine import (
    any_flag, bands, cat, coord, flag, num, numbers, parse_date, text)

YEAR = 2010
RAW_DIR = 'MZMV2010_geo/4_DB_csv/db_csv'
RAW_FILES = {'hh': 'haushalte', 'hp': 'haushaltspersonen', 'zp': 'zielpersonen',
             'fz': 'fahrzeuge', 'wg': 'wege', 'et': 'etappen',
             'dj': 'reisen1a', 'oj': 'reisen1b'}
READ_KWS = dict(sep=None, engine='python', encoding='latin-1')
XY_CRS = 'EPSG:4326'

SEX = {1: 'male', 2: 'female'}

# f51300: same codes as F510 in 2005
MODE = {**codebook_2005.MODE, 2: 'anybike'}     # Velo, incl. e-bikes
PASSENGER, DRIVER = codebook_2005.PASSENGER, codebook_2005.DRIVER

PURPOSE = codebook_2005.PURPOSE                 # f52900 / wzweck1
DIRECTION = codebook_2005.DIRECTION             # wzweck2
INCOME = codebook_2005.INCOME                   # F20601
VEHICLE_TYPE = codebook_2005.VEHICLE_TYPE       # fahrzeugart
CAR_SOURCE = codebook_2005.CAR_SOURCE           # f51310a

# f30400: Treibstoff Auto
FUEL = {
    1: 'petrol',                # Benzin
    2: 'diesel',                # Diesel
    3: 'alternative',           # Hybrid / E85 / Gas
    4: 'other',                 # andere
}

# mf_getri (registry): Getriebe (codes 5-8 from 2015)
TRANSMISSION = {
    1: 'automatic',                 # A. Automatik
    2: 'manual',                    # M. Manuell
    3: 'hydrostatic',               # H. Hydrostatisch
    4: 'continuously_variable',     # S. Stufenlos
    5: 'automated_manual',          # AM. automatisch, mit integriertem mechanischem Getriebe
    6: 'automated_manual',          # MS. mechanisch, selbständig kuppelnd
    7: 'automated_manual',          # MA. mechanisch, selbständig kuppelnd
    8: 'continuously_variable',     # SM. stufenlos, mit virtuellen Schaltstufen
}

# f51700 (legs) / f51700 (trips; f51700_weg from 2015): Freizeitaktivität am Etappenziel
LEISURE = {
    1: 'visit',                 # Besuche (Verwandte, Bekannte, Freunde)
    2: 'gastronomy',            # Gastronomiebesuch
    3: 'sport_active',          # Aktiver Sport (Fussball, Jogging etc.)
    4: 'hiking',                # Wanderung
    5: 'cycling',               # Velofahrt
    6: 'sport_passive',         # Passiver Sport (Matchbesuch etc.)
    7: 'outdoor_nonsport',      # Nicht-sportliche Aussenaktivität (z.B. Spaziergang)
    8: 'wellness',              # Medizin / Wellness / Fitness
    9: 'culture',               # Kulturveranstaltungen, Freizeitanlagen
    10: 'unpaid_work',          # Unbezahlte Arbeit
    11: 'club',                 # Vereinstätigkeit
    12: 'excursion',            # Ausflug, Ferien
    13: 'religion',             # Religion (Kirche, Friedhof, Pilgerfahrt)
    14: 'home_activity_away',   # Häusliche Freizeitaktivitäten auswärts
    15: 'picnic',               # Essen ohne Gastronomiebesuch (Pick-Nick)
    16: 'shopping_stroll',      # Einkaufsbummel / Shopping
    17: 'round_trip',           # Rundreise (Auto-/Motorrad-/Zugsfahrt)
    18: 'other',                # Anderes (2010, 2015)
    22: 'multiple',             # mehrere Aktivitäten (2010)
    90: 'other',                # Anderes (2021)
}

# S_QAL / Z_QAL (kept up to 2021)
GEOCODE_LEVEL = {
    1: 'address',               # A) Hausnummer, BUR, GWR, POIS, Haltestelle
    2: 'street',                # B) Strasse
    3: 'municipality',          # C) Gemeinde, PLZ, Ort
    4: 'region',                # D) NUTS-3 (angrenzendes Ausland)
    5: 'country',               # E) Land
}

EXTRA_MODULE = codebook_2005.EXTRA_MODULE       # dmod

# f60800_01 / f70700_01: Zweck der Tagesreise / Reise mit Übernachtung (kept to 2021)
JOURNEY_PURPOSE = {
    1: 'education',             # Ausbildung, Schul-/Studienreise, Lager
    2: 'shopping',              # Shopping
    3: 'medical',               # Medizinische Behandlung (Klinikbesuch etc.)
    4: 'business',              # Geschäftsreise
    5: 'visit',                 # Besuche (Verwandte, Bekannte)
    6: 'gastronomy',            # Gastronomiebesuch
    7: 'sport_active',          # Aktiver Sport
    8: 'hiking',                # Wanderung
    9: 'cycling',               # Velofahrt
    10: 'sport_passive',        # Passiver Sport
    11: 'outdoor_nonsport',     # Nicht-sportliche Aussenaktivitäten
    12: 'culture',              # Kulturveranstaltung, Freizeitanlagen
    13: 'excursion_holiday',    # Ausflug/Ferien
    14: 'religion',             # Religion (Kirche, Friedhof, Pilgerfahrt)
    15: 'accompany_business',   # Begleitung auf Geschäftsreise
    16: 'accompany_private',    # Begleitung auf Privatreise
    17: 'round_trip',           # Rundreise
    18: 'other',                # Anderes
}

# f61800 / f71800: Regelmässigkeit (kept to 2021)
REGULARITY = {
    1: 'weekly',                # Mind. einmal pro Woche
    2: 'monthly',               # Mind. einmal pro Monat
    3: 'yearly',                # Mind. einmal pro Jahr
    4: 'less_often',            # Seltener
    5: 'first_time',            # Zum ersten Mal aufgesucht
}

# Raw variables of the journey tables, 2010-2021: day (Reisen1a / Tagesreisen), overnight
# (Reisen1b / Reisen mit Übernachtung). From 2015 the distances come in two versions.
_JOURNEY_VARS = {
    'day': dict(selected='trenr', date='reisedatum', purpose='f60800_01', mode='f60900_01',
                same_back='f61300', mode_back='f61401', companions='f61500', dist='f61600',
                dist_domestic='f61700', regularity='f61800', has_places='f60500', orig='trs',
                dest='trz', places='zh'),
    'overnight': dict(selected='renr', date='reisestart', purpose='f70700_01', mode='f70801',
                      same_back='f71300', mode_back='f71400_01', companions='f71500',
                      dist='f71600', dist_domestic='f71700', regularity='f71800',
                      has_places='f70550', orig='rs', dest='rz', places='rs'),
}
PLACE_TYPE = {'day': 'stop', 'overnight': 'overnight'}

# hhtyp: Typ des Privathaushalts
HH_TYPE = {
    100: 'single',              # Einpersonenhaushalt
    210: 'couple_no_kids',      # Paare ohne Kinder
    220: 'couple_kids',         # Paare mit Kind(ern)
    230: 'single_parent',       # Einelternhaushalte mit Kind(ern)
    240: 'single_parent',       # Erwachsene Person mit Elternteil (merged into 230 from 2015)
    300: 'other',               # Nichtfamilienhaushalte
}

# HAUSB: Höchste abgeschlossene Ausbildung
EDUCATION = {
    1: 'none',                          # hat keine Schule besucht
    2: 'none',                          # obligatorische Schule nicht abgeschlossen
    3: 'compulsory',                    # nur obligatorische Schule
    4: 'compulsory',                    # 1-jährig: 10. Schuljahr, Vorlehre, Brückenangebote
    5: 'upper_secondary_vocational',    # 2-jährige berufliche Grundbildung (EBA)
    6: 'upper_secondary_vocational',    # 2-jährige Vollzeitberufsschule, Handelsschule
    7: 'upper_secondary_general',       # 2-3-jährig allgemeinbildend (DMS, FMS)
    8: 'upper_secondary_vocational',    # 3-4-jährige Berufslehre
    9: 'upper_secondary_vocational',    # 3-4-jährige Vollzeitberufsschule
    10: 'upper_secondary_general',      # Lehrkräfte-Seminar
    11: 'upper_secondary_general',      # Gymnasiale Maturität
    12: 'upper_secondary_general',      # Berufs- oder Fachmaturität
    13: 'tertiary_professional',        # Höhere Berufsbildung (eidg. Fachausweis / Diplom)
    14: 'tertiary_academic',            # Höhere Fachschule (HF) Technik / Wirtschaft
    15: 'tertiary_academic',            # Höhere Fachschule HWV, HFG, HFS, HTL
    16: 'tertiary_academic',            # Fachhochschule, Pädagogische Hochschule
    17: 'tertiary_academic',            # Universität, ETH
}

# ESTATUS: Erwerbsstatus (10-stufig, BFS labour-force concept; kept up to 2021)
EMPLOYMENT = {
    1: 'employed',              # Selbständige
    2: 'employed',              # Mitarbeitendes Familienmitglied
    3: 'employed',              # Arbeitnehmer/in
    4: 'in_education',          # Lehrling (as in the self-assessed years)
    5: 'unemployed',            # Erwerbslose
    6: 'in_education',          # Nichterwerbsperson in Aus-/Weiterbildung
    7: 'retired',               # Nichterwerbsperson im Ruhestand
    8: 'other_inactive',        # Invalide Nichterwerbsperson
    9: 'homemaker',             # Hausfrau / Hausmann
    10: 'other_inactive',       # Andere Nichterwerbsperson
}

# f50100a: Wetter am Stichtag, first mention (kept up to 2021)
WEATHER = {
    1: 'sunny',                 # Sonnig/schön
    2: 'partly_cloudy',         # Leicht bewölkt
    3: 'cloudy',                # Bewölkt/stark bewölkt
    4: 'foggy',                 # Neblig/Nebel
    5: 'rainy',                 # Regen
    6: 'snowy',                 # Schnee
    7: 'unstable',              # Stark veränderlich/unstabil
    8: 'hot',                   # Warm/heiss
    9: 'cold',                  # Kalt/eisig
}


def households(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hh = raw['hh']
    income_min, income_max = bands(hh['f20601'], INCOME)
    return pd.DataFrame({
        'hhnr': hh['hhnr'],
        'weight': num(hh['wm']),
        'home_x': coord(hh['w_x']),
        'home_y': coord(hh['w_y']),
        'home_geocode_level': cat(hh['w_qal'], GEOCODE_LEVEL),
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
        'n_bikes_any': num(hh['f32200']),
        'has_second_home': flag(hh['f10700']),
    })


def people(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zp = raw['zp']
    passes = {c: flag(zp[f'f41600{c}']) for c in 'abcdefg'}
    # In pass_season, not asked (-99; Gleis 7 only at age 16-25) means not held.
    season = {c: flag(zp[f'f41600{c}'], no=(2, -99)) for c in 'defg'}
    return pd.DataFrame({
        'hhnr': zp['hhnr'],
        'person_no': zp['zielpnr'],
        'weight': num(zp['wp']),
        'age': num(zp['alter']),
        'sex': cat(zp['gesl'], SEX),
        'survey_date': parse_date(zp['vstag']),
        'weekday': num(zp['tag']),
        'is_mobile': flag(zp['f50200']),
        'license_car': flag(zp['f20400a']),
        'license_motorbike': flag(zp['f20400b']),
        'pass_ga': any_flag(passes['b'], passes['c']),         # GA 1. / 2. Klasse
        'pass_halbtax': passes['a'],
        'pass_regional': passes['d'],                           # Verbund-Abo
        'pass_season': any_flag(*season.values()),              # Verbund, Strecke, Gleis 7, anderes
        'education': cat(zp['hausb'], EDUCATION),
        'employment': cat(zp['estatus'], EMPLOYMENT),
        'employed_fulltime': flag(zp['f40900'], yes=(1,), no=(2, 3)),
        'weather': cat(zp['f50100a'], WEATHER),
        **journey_counts(zp),
    })


def household_members(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hp = raw['hp']
    return pd.DataFrame({
        'hhnr': hp['hhnr'],
        'member_no': hp['hpnr'],
        'person_no': hp['zielpnr'],                 # 0 = not a target person
        'weight': num(hp['wm']),
        'age': num(hp['alter']),
        'sex': cat(hp['gesl'], SEX),
        'license_car': flag(hp['f20400a']),
        'license_motorbike': flag(hp['f20400b']),
    })


def vehicles(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fz = raw['fz']
    engine_cc = num(fz['f30500_31400'])
    km = num(fz['f30900_31700'])
    return pd.DataFrame({
        'hhnr': fz['hhnr'],
        'weight': num(fz['wm']),
        'vehicle_type': cat(fz['fahrzeugart'], VEHICLE_TYPE),
        'vehicle_no': fz['fznum'],
        'registration_year': num(fz['f30600_31500']),
        'registration_month': num(fz['f30601_31501']),
        'odometer_km': num(fz['f30800_31600']),
        'km_per_year': km,                          # last 12 months
        'km_per_year_abroad': num(fz['f31000_31800']).mask(km == 0, 0),
        'engine_cc_min': engine_cc,
        'engine_cc_max': engine_cc,
        'fuel': cat(fz['f30400'], FUEL),
        'main_driver_nos': numbers(fz['f30700_hpnr1'], fz['f30700_hpnr2'],
                                   na=(16,)),                       # 16 = other person
        **registry(fz),
    })


def registry(fz: pd.DataFrame) -> dict[str, pd.Series]:
    """Vehicle-registry (MOFIS) fields, empty for vehicles not matched; 2015 adds more."""
    return {
        'in_registry': flag(fz['mofis']),
        'weight_total_kg': num(fz['mf_gege1']),
        'cylinders': num(fz['mf_zylin']),
        'transmission': cat(fz['mf_getri'], TRANSMISSION),
        'energy_label': text(fz['mf_eekat']),
        'emission_code': text(fz['mf_emcod']),
    }


def trips(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wg = raw['wg']
    return pd.DataFrame({
        'hhnr': wg['hhnr'],
        'person_no': wg['zielpnr'],
        'trip_no': wg['wegnr'],
        'weight': num(wg['wp']),
        'dep_min': num(wg['f51100']),
        'arr_min': num(wg['f51400']),
        'duration_min': num(wg['dauer2']),
        'in_motion_min': num(wg['dauer1']),
        'dist_km': num(wg['w_rdist']),
        'dist_estimated_km': num(wg['w_dist_subj']),
        'n_legs': num(wg['w_etappen']),
        'has_pseudo_leg': flag(wg['pseudo'], yes=(2,), no=(1,)),
        'purpose': cat(wg['wzweck1'], PURPOSE),
        'direction': cat(wg['wzweck2'], DIRECTION),
        'leisure_activity': cat(wg['f51700'], LEISURE),
        **endpoints(wg),
    })


def legs(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    et = raw['et']
    return pd.DataFrame({
        'hhnr': et['hhnr'],
        'person_no': et['zielpnr'],
        'trip_no': et['wegnr'],
        'leg_no': et['etnr'],
        'weight': num(et['wp']),
        'dep_min': num(et['f51100']),
        'arr_min': num(et['f51400']),
        'duration_min': num(et['e_dauer']),
        'dist_km': num(et['rdist']),
        'dist_estimated_km': num(et['f51500']),
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
    return journeys(raw['dj'], 'day', MODE, PASSENGER, DRIVER)


def overnight_journeys(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return journeys(raw['oj'], 'overnight', MODE, PASSENGER, DRIVER)


def journey_places(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stops of the day journeys (ZH1-3) and further overnight places (RS1-3), 2010-2021."""
    return pd.concat([places(raw['dj'], 'day'), places(raw['oj'], 'overnight')],
                     ignore_index=True)


def places(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One row per place named (up to 3 per detailed journey)."""
    df = df[num(df['reisenr']) > 0]
    raw_prefix = _JOURNEY_VARS[kind]['places']
    out = []
    for no in (1, 2, 3):
        p = pd.DataFrame(endpoints(df, (('place', f'{raw_prefix}{no}'),)))
        named = p[['place_x', 'place_municipality', 'place_in_switzerland']].notna().any(axis=1)
        out.append(p[named].assign(
            hhnr=df['hhnr'], person_no=df.get('zielpnr', 1),
            journey_no=df['reisenr'], place_no=no, place_type=PLACE_TYPE[kind]))
    return pd.concat(out)


def journey_counts(zp: pd.DataFrame) -> dict[str, pd.Series]:
    """Module assigned and journeys reported (2010-2021)."""
    return {
        'extra_module': cat(zp['dmod'], EXTRA_MODULE),
        'n_day_journeys': num(zp['f60100']),
        'n_overnight_journeys': num(zp['f70100']),
    }


def journeys(df: pd.DataFrame, kind: str, mode: dict, passenger: tuple, driver: tuple,
             dist_version: str = '') -> pd.DataFrame:
    """Day or overnight journeys (`kind`), 2010-2021: one row per journey reported, details
    for the ones picked (TRENR / RENR set). `dist_version` 'b' (2015 on) takes the distances
    the BFS imputed where the respondent did not know. Rows the BFS marks invalid (journey
    no. 0, 2015 on) are dropped."""
    v = _JOURNEY_VARS[kind]
    valid = num(df['reisenr']) > 0
    if not valid.all():
        logging.info(f"  {kind} journeys: dropping {int((~valid).sum()):,} rows marked invalid")
    df = df[valid]
    mode_main = cat(df[v['mode']], mode)
    same_back = flag(df[v['same_back']])
    ends = endpoints(df, (('orig', v['orig']), ('dest', v['dest'])))
    dist = num(df[v['dist'] + dist_version])
    # Asked for journeys abroad only.
    domestic = num(df[v['dist_domestic'] + dist_version])
    domestic = domestic.mask(domestic.isna() & (ends['dest_in_switzerland'] == 1), dist)
    out = pd.DataFrame({
        'hhnr': df['hhnr'],
        'person_no': df.get('zielpnr', 1),
        'journey_no': df['reisenr'],
        'weight': num(df['wp']),
        'is_detailed': (num(df[v['selected']]) > 0).astype(float),
        'date': parse_date(df[v['date']]),
        'purpose': cat(df[v['purpose']], JOURNEY_PURPOSE),
        'mode_main': mode_main,
        'is_passenger': flag(df[v['mode']], yes=passenger, no=driver, na=mode),
        'same_mode_back': same_back,
        'mode_back': cat(df[v['mode_back']], mode).mask(same_back == 1, mode_main),
        'dist_km': dist,
        'dist_domestic_km': domestic,
        'regularity': cat(df[v['regularity']], REGULARITY),
        'has_places': flag(df[v['has_places']]),
        **ends,
    })
    if v['companions'] in df:                       # not asked in 2021
        out['n_companions'] = num(df[v['companions']])
    if kind == 'day':
        out['duration_h'] = num(df['f60700'])
    else:
        out['nights'] = num(df['f70500'])
    return out


def endpoints(df: pd.DataFrame,
              prefixes: tuple[tuple[str, str], ...] = (('orig', 's'), ('dest', 'z'))
              ) -> dict[str, pd.Series]:
    """Location columns, 2010-2021, as (canonical prefix, raw prefix) pairs: trips and legs
    S_* / Z_*, journeys and their places their own (WGS84, country as BFS state code with
    8100 = Switzerland)."""
    out = {}
    for prefix, raw_prefix in prefixes:
        country = num(df[f'{raw_prefix}_lnd'])
        in_ch = (country == 8100).astype(float).where(country.notna())
        out |= {
            f'{prefix}_x': coord(df[f'{raw_prefix}_x']),
            f'{prefix}_y': coord(df[f'{raw_prefix}_y']),
            f'{prefix}_postcode': num(df[f'{raw_prefix}_plz']).where(in_ch != 0),
            f'{prefix}_municipality': num(df[f'{raw_prefix}_bfs']).where(in_ch != 0),
            f'{prefix}_in_switzerland': in_ch,
        }
        if f'{raw_prefix}_qal' in df:               # trips, legs (journeys 2015 only)
            out[f'{prefix}_geocode_level'] = cat(df[f'{raw_prefix}_qal'], GEOCODE_LEVEL)
    return out
