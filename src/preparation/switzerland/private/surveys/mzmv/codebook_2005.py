"""
MTMC 2005 (Mikrozensus zum Verkehrsverhalten 2005) → canonical schema.

Source documentation: `ED_2007-01-24_TD_PT_Datenbankstruktur_inkl_korrigierte_Distanzen.xls`,
`ED_2007-01-24_TD_E1_MZ05_Datenbankbeschreibung.pdf`, `EA-2012-06-18_Hinweis_Vergleichbarkeit`.

Year specifics:
  - Up to two target persons per household (ZIELPNR: one in households up to three people
    aged 6+, two above).
  - Coordinates are CH1903/LV03 integer metres (the codebook does not say; the axis order is
    detected from the values).
  - `rdist` is the respondent's estimate corrected with the 2010 routing factors (BFS 2012);
    no routed distances and no route geometries.
  - Trips were built by the BFS from legs: a trip continues while the purpose stays the same,
    is not 'transfer', and the gap is ≤ 1 h.
  - Bikes: e-bikes are not told apart; they were still rare → 'rbike'.
  - Transit passes are one combined answer (F26), incl. some combinations.
  - Hiking and cycling are not separate leisure activities (F5202).
  - Vehicles: cars and motorbikes numbered per type by km driven; fuel only petrol / diesel /
    other; one main driver per car. Car legs say which household car (F511A, only asked in
    households with more than one car).
  - Journeys (modules 1a / 1b, DMOD): first year. One purpose (sport not split; 'Kombination'
    → 'multiple'), the mode covering the longest distance of the whole journey, whether it
    was used back, no return mode and no domestic distance. Regularity has an extra
    'several times a week' (→ 'weekly'). Places are geocoded free text: LV03 metres in
    Switzerland, WGS84 degrees abroad (undocumented; converted).
"""

import logging

import numpy as np
import pandas as pd
from pyproj import Transformer

from preparation.switzerland.private.surveys.mzmv.geo import CRS_LV03
from preparation.switzerland.private.surveys.mzmv.engine import (
    bands, cat, flag, num, numbers, parse_date)

YEAR = 2005
RAW_DIR = 'MZ2005_geo/4_DB_ASCII(Sep_TAB)/Dateien'
RAW_FILES = {'hh': 'haushalte', 'hp': 'haushaltspersonen', 'zp': 'zielpersonen',
             'fz': 'autos_motos', 'wg': 'wege', 'et': 'etappen',
             'dj': 'reisen_1a', 'oj': 'reisen_1b'}
READ_KWS = dict(sep='\t', encoding='latin-1')
XY_CRS = 'EPSG:21781'
# Journeys mix LV03 (Switzerland) and degrees (abroad); `_wgs84` brings them to degrees.
XY_CRS_BY_TABLE = dict.fromkeys(('day_journeys', 'overnight_journeys', 'journey_places'),
                                'EPSG:4326')

SEX = {1: 'male', 2: 'female'}

# F510: Welches Verkehrsmittel haben Sie benutzt? (same codes as f51300 in 2010)
MODE = {
    1: 'walk',                  # zu Fuss
    2: 'rbike',                 # Velo
    3: 'motorbike_small',       # Mofa, Motorfahrrad
    4: 'motorbike_small',       # Kleinmotorrad (ab 16 J.)
    5: 'motorbike',             # Motorrad als Fahrer
    6: 'motorbike',             # Motorrad als Mitfahrer
    7: 'car',                   # Auto als Fahrer
    8: 'car',                   # Auto als Mitfahrer
    9: 'transit_rail',          # Bahn
    10: 'transit_bus',          # Postauto
    11: 'transit_bus',          # Bus
    12: 'transit_tram',         # Tram
    13: 'taxi',                 # Taxi
    14: 'coach',                # Reisecar
    15: 'truck',                # Lastwagen
    16: 'boat',                 # Schiff
    17: 'plane',                # Flugzeug
    18: 'aerialway',            # Zahnradbahn, Standseilbahn, Seilbahn, Sessellift, Skilift
    19: 'micro',                # Fahrzeugähnliche Geräte
    20: 'other',                # Anderes
}
PASSENGER, DRIVER = (6, 8), (5, 7)

# F516 (legs) and wzweck1 (trips); f52900 / wzweck1 keep these codes up to 2021.
PURPOSE = {
    1: 'mode_transfer',         # Umsteigen, Verkehrsmittelwechsel, Auto abstellen
    2: 'work_commute',          # Arbeiten
    3: 'education',             # Ausbildung, Schule
    4: 'errands_groceries',     # Einkaufen
    5: 'errands_services',      # Besorgungen und Inanspruchnahme von Dienstleistungen
    6: 'work_business',         # Geschäftliche Tätigkeit
    7: 'work_business',         # Dienstfahrt
    8: 'leisure',               # Freizeitaktivität
    9: 'accompany',             # Begleitweg (nur Kinder)
    10: 'accompany',            # Begleitweg / Serviceweg (Andere, z.B. Behinderte)
    11: 'return',               # Rückkehr nach Hause bzw. auswärtige Unterkunft
    12: 'other',                # Anderes
    13: 'border_crossing',      # Grenzübergang
}

# wzweck2 (kept up to 2021). The 2005 description swaps codes 2 and 3 in its text; the value
# labels (used here) agree with the 2015 description.
DIRECTION = {
    1: 'outbound',              # Hinweg
    2: 'return',                # Nachhauseweg
    3: 'round',                 # Weg von zu Hause nach Hause
}

# F5202: Freizeitaktivität, aggregated (order differs from f51700 used from 2010 on)
LEISURE = {
    1: 'visit',                 # Besuch
    2: 'gastronomy',            # Gastronomiebesuch
    3: 'outdoor_nonsport',      # Nicht sportliche Aussenaktivitäten
    4: 'sport_active',          # Aktiver Sport
    5: 'sport_passive',         # Passiver Sport
    6: 'wellness',              # Medizin, Wellness
    7: 'culture',               # Kulturveranstaltungen, Freizeitanlagen
    8: 'club',                  # Vereine
    9: 'excursion',             # Ausflug, Ferien
    10: 'unpaid_work',          # Unbezahlte Arbeit
    11: 'shopping_stroll',      # Einkaufsbummel
    12: 'religion',             # Kirche, Friedhof
    13: 'home_activity_away',   # Freizeitaktivitäten im Haus (z.B. bei Nachbarn)
    14: 'picnic',               # Essen ohne Gastronomiebesuch
    15: 'other',                # Übriges
    16: 'multiple',             # Mehrere Aktivitäten
    17: None,                   # Keine Angaben
    18: 'accompany',            # Begleitung als Freizeit
    19: None,                   # Rückkehr als Freizeit
}

# S_QAL / Z_QAL: Erreichte Qualitätsstufe
GEOCODE_LEVEL = {
    1: 'address',               # A) Hausnummer, BUR, GWR, POIS, Haltestelle
    2: 'street',                # B) Strasse
    3: 'municipality',          # C) Gemeinde, PLZ, Land
    4: None,                    # D) Keine Information (trips only)
}

# DMOD: Ausgewähltes Modul (kept to 2021)
EXTRA_MODULE = {
    0: None,                    # Modul 3 übersprungen: Proxy-Interview, minderjährig (2015 on)
    1: 'day_journeys',          # Modul 1a Tagesreisen
    2: 'overnight_journeys',    # Modul 1b Reisen mit Übernachtung
    3: 'slow_modes',            # Modul 2 Langsamverkehr und Bewegung
    4: 'opinions',              # Modul 3 Verkehrspolitische Einstellungen
}

# F69 / F78: Welchen Zweck hat diese (Tages)reise gehabt?
JOURNEY_PURPOSE = {
    1: 'visit',                 # Besuche (Verwandte, Bekannte)
    2: 'sport',                 # Sportaktivität
    3: 'round_trip',            # Rundreise
    4: 'outdoor_nonsport',      # Nicht sportliche Aussenaktivitäten
    5: 'culture',               # Kulturveranstaltung, Freizeitanlagen
    6: 'gastronomy',            # Gastronomiebesuch
    7: 'unpaid_work',           # unbezahlte Arbeit
    8: 'excursion_holiday',     # Ferien
    9: 'medical',               # Medizinische Behandlung
    10: 'religion',             # Religion/Pilgerfahrt
    11: 'business',             # Geschäftsreise
    12: 'accompany_business',   # Geschäftsreise als Begleiter
    13: 'other',                # Anderes
    20: 'multiple',             # Kombination
}

# F615 / F714: Wie regelmässig reisen Sie an den Ort von der Reise x?
REGULARITY = {
    1: 'weekly',                # Mehrmals pro Woche
    2: 'weekly',                # mind. einmal pro Woche
    3: 'monthly',               # mind. einmal pro Monat
    4: 'yearly',                # mind. einmal pro Jahr
    5: 'first_time',            # zum ersten Mal aufgesucht
    6: 'less_often',            # seltener
}

# Raw variables of the journey tables: day (Reisen_1a), overnight (Reisen_1b).
_JOURNEY_VARS = {
    'day': dict(selected='trenr', date='f64', purpose='f69', mode='f610', same_back='f611',
                companions='f613', dist='f614', regularity='f615', orig='trs', dest='trz'),
    'overnight': dict(selected='renr', date='f74', purpose='f78', mode='f79', same_back='f710',
                      companions='f712', dist='f713', regularity='f714', orig='rs', dest='rz'),
}

# F21: Wie lässt sich Ihr Haushalt beschreiben?
HH_TYPE = {
    1: 'single',                # Einpersonenhaushalt
    2: 'single_parent',         # Einelternhaushalt mit Kindern
    3: 'couple_kids',           # (Ehe-) Paar-Haushalt mit Kindern
    4: 'couple_no_kids',        # (Ehe-) Paar-Haushalt ohne Kinder
    5: 'other',                 # Wohngemeinschaft
    6: 'other',                 # Untermiete
    7: 'other',                 # übrige Haushalte (Anstalten, Kollektivhaushalte)
}

# F101 (and f20601 up to 2021): Brutto-Monatseinkommen des Haushalts, CHF
INCOME = {
    1: (0, 1999),               # unter 2000
    2: (2000, 4000),
    3: (4001, 6000),
    4: (6001, 8000),
    5: (8001, 10000),
    6: (10001, 12000),
    7: (12001, 14000),
    8: (14001, 16000),
    9: (16001, None),           # mehr als 16000
}

# F43: Welche Ausbildung haben Sie zuletzt abgeschlossen?
EDUCATION = {
    1: 'none',                          # Keine Ausbildung abgeschlossen
    2: 'compulsory',                    # Obligatorische Schule
    3: 'upper_secondary_vocational',    # Berufslehre
    4: 'upper_secondary_vocational',    # Vollzeitberufsschule (z.B. Handelsschule)
    5: 'upper_secondary_general',       # Maturitätsschule, Primarlehrerausbildung
    6: 'tertiary_professional',         # Höhere Berufsausbildung (Meister, eidg. Fachausweis)
    7: 'tertiary_academic',             # Technikerschule, höhere Fachschule, Fachhochschule
    8: 'tertiary_academic',             # Universität, Hochschule
    9: None,                            # keine Angabe
}

# F44: Sind Sie berufstätig? (same codes as 1994 HP05 and 2000 F50400)
EMPLOYMENT = {
    1: 'employed',              # Voll erwerbstätig
    2: 'employed',              # Teilzeit, eine Stelle
    3: 'employed',              # Teilzeit, zwei oder mehr Stellen
    4: 'unemployed',            # nicht erwerbstätig, auf Stellensuche
    5: 'unemployed',            # nicht erwerbstätig, künftige Stelle zugesichert
    6: 'other_inactive',        # nicht erwerbstätig und nicht suchend
    7: 'homemaker',             # Hausarbeit im eigenen Haushalt
    8: 'in_education',          # In Ausbildung (Schule, Studium, Lehre)
    9: 'retired',               # Rentner/in, pensioniert
    10: 'other_inactive',       # Andere Situation (z.B. chronisch krank)
    11: None,                   # Keine Angabe / weiss nicht
}

# F51: Wie ist das Wetter gewesen?
WEATHER = {
    1: 'snowy',                 # Schneefall
    2: 'rainy',                 # Regen
    3: 'sunny',                 # schön und warm
    4: 'sunny',                 # schön und kalt
    5: 'sunny',                 # schön
    6: 'hot',                   # warm
    7: 'cold',                  # kalt
    8: 'unstable',              # unstabil (stark veränderlich)
    9: 'cloudy',                # übriges (neblig, bewölkt)
    10: None,                   # weiss nicht
}

# F26: Abonnement(e) des öffentlichen Verkehrs — one answer:
#   1 Halbtax, 2 GA 1. Kl., 3 GA 2. Kl., 4 Wochenabo, 5 Monatsabo, 6 Jahresabo, 7 Streckenabo,
#   8 Gleis 7, 9 Juniorkarte, 10 andere, 11 keines, 12-17 Halbtax and (Wochen-, Monats-,
#   Jahres-, Strecken-, Gleis-7-, anderes Abo), 20 übrige Kombinationen (GA, half-fare unknown)
_SEASON = (4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20)
_HALBTAX = (1, 12, 13, 14, 15, 16, 17)

# FAHRZ (fahrzeugart from 2010): Fahrzeug Auto oder Motorrad
VEHICLE_TYPE = {1: 'car', 2: 'motorbike'}

# F32A: Welche Art von Treibstoff benutzen Sie für das Auto x?
FUEL = {
    1: 'petrol',                # Benzin
    2: 'diesel',                # Diesel
    3: 'other',                 # Andere
    4: None,                    # Weiss nicht
}

# F511 (f51310a from 2010): Was für ein Auto haben Sie benutzt?
CAR_SOURCE = {
    1: 'household',             # Auto im Haushalt
    2: 'company',               # Firmenauto, Dienstwagen
    3: 'rental',                # Mietauto
    4: 'carsharing',            # Carsharing-Auto, Autoteilen
    5: 'other',                 # Anderes Auto
}

# S_LAND / Z_LAND are country names (no value labels in the codebook).
_SWITZERLAND = {'schweiz', 'suisse', 'svizzera', 'switzerland', 'ch', '8100'}


def households(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hh = raw['hh']
    hh_size = num(hh['f22a']).mask(num(hh['f21']) == 1, 1)   # only asked for F21 in 2-6
    income_min, income_max = bands(hh['f101'], INCOME)
    return pd.DataFrame({
        'hhnr': hh['hhnr'],
        'weight': num(hh['wm']),
        'home_x': num(hh['w_x']),
        'home_y': num(hh['w_y']),
        'home_geocode_level': cat(hh['w_qal'], GEOCODE_LEVEL),
        'home_postcode': num(hh['w_plz']),
        'home_municipality': num(hh['w_bfs']),
        'home_canton_no': num(hh['w_kanton']),
        'n_people': hh_size,
        'hh_type': cat(hh['f21'], HH_TYPE),
        'income_min_chf': income_min,
        'income_max_chf': income_max,
        'n_cars': num(hh['f31']),
        'n_motorbikes': num(hh['f310']),
        'n_small_motorbikes': num(hh['f316']),
        'n_mopeds': num(hh['f318']),
        'n_bikes_any': num(hh['f319']),
        'n_car_parking': num(hh['f39']),
        'has_second_home': flag(hh['f19a']),
    })


def people(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zp = raw['zp']
    return pd.DataFrame({
        'hhnr': zp['hhnr'],
        'person_no': zp['zielpnr'],
        'weight': num(zp['wp']),
        'age': num(zp['f23b']),
        'sex': cat(zp['f23c'], SEX),
        'survey_date': parse_date(zp['vstag']),
        'weekday': num(zp['tag']),
        'is_mobile': flag(zp['f55']),
        'license_car': flag(zp['f24']),
        'license_motorbike': flag(zp['f25']),
        **_passes(zp['f26']),
        'education': cat(zp['f43'], EDUCATION),
        'employment': cat(zp['f44'], EMPLOYMENT),
        'employed_fulltime': flag(zp['f44'], yes=(1,), no=(2, 3), na=EMPLOYMENT),
        'weather': cat(zp['f51'], WEATHER),
        'extra_module': cat(zp['dmod'], EXTRA_MODULE),
        'n_day_journeys': num(zp['f61']),
        'n_overnight_journeys': num(zp['f71']),
    })


def household_members(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hp = raw['hp']
    return pd.DataFrame({
        'hhnr': hp['hhnr'],
        'member_no': hp['hpnr'],
        'person_no': hp['zielpnr'],                 # 0 = not a target person
        'weight': num(hp['wm']),
        'age': num(hp['f23b']),
        'sex': cat(hp['f23c'], SEX),
        'license_car': flag(hp['f24']),
        'license_motorbike': flag(hp['f25']),
        **_passes(hp['f26']),
    })


def vehicles(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fz = raw['fz']
    engine_cc = num(fz['f33'])
    km = num(fz['f37'])
    return pd.DataFrame({
        'hhnr': fz['hhnr'],
        'weight': num(fz['wm']),
        'vehicle_type': cat(fz['fahrz'], VEHICLE_TYPE),
        'vehicle_no': fz['fznum'],                  # per type, by km driven
        'registration_year': num(fz['f34']),
        'registration_month': num(fz['f34a'], na=(13,)),     # 13 = don't know
        'odometer_km': num(fz['f36']),
        'km_per_year': km,                          # last 12 months
        'km_per_year_abroad': num(fz['f38']).mask(km == 0, 0),
        'engine_cc_min': engine_cc,
        'engine_cc_max': engine_cc,
        'fuel': cat(fz['f32a'], FUEL),
        'main_driver_nos': numbers(fz['f35_hpnr'], na=(16,)),    # cars only; 16 = other
    })


def trips(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wg = raw['wg']
    return pd.DataFrame({
        'hhnr': wg['hhnr'],
        'person_no': wg['zielpnr'],
        'trip_no': wg['wegnr'],
        'weight': num(wg['wp']),
        'dep_min': num(wg['f58']),
        'arr_min': num(wg['f514']),
        'duration_min': num(wg['dauer2']),
        'in_motion_min': num(wg['dauer1']),
        'dist_km': num(wg['w_rdist']),
        'dist_estimated_km': num(wg['w_dist_subj']),
        'n_legs': num(wg['w_etappen']),
        'has_pseudo_leg': flag(wg['pseudo'], yes=(2,), no=(1,)),
        'purpose': cat(wg['wzweck1'], PURPOSE),
        'direction': cat(wg['wzweck2'], DIRECTION),
        'leisure_activity': cat(wg['f5202'], LEISURE),
        **_endpoints(wg),
    })


def legs(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    et = raw['et']
    return pd.DataFrame({
        'hhnr': et['hhnr'],
        'person_no': et['zielpnr'],
        'trip_no': et['wegnr'],
        'leg_no': et['etnr'],
        'weight': num(et['wp']),
        'dep_min': num(et['f58']),
        'arr_min': num(et['f514']),
        'duration_min': num(et['e_dauer']),
        'dist_km': num(et['rdist']),
        'dist_estimated_km': num(et['f515']),
        'mode': cat(et['f510'], MODE),
        'is_passenger': flag(et['f510'], yes=PASSENGER, no=DRIVER, na=MODE),
        'car_occupancy': num(et['f512']),
        'car_source': cat(et['f511'], CAR_SOURCE),
        'car_no': num(et['f511a']),                 # only asked if more than one car
        'purpose': cat(et['f516'], PURPOSE),
        'leisure_activity': cat(et['f5202'], LEISURE),
        'is_abroad': flag(et['e_ausland']),
        'is_pseudo': flag(et['pseudo'], yes=(2,), no=(1,)),
        **_endpoints(et),
    })


def day_journeys(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return _journeys(raw['dj'], 'day').assign(duration_h=num(raw['dj']['f66']))


def overnight_journeys(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return _journeys(raw['oj'], 'overnight').assign(nights=num(raw['oj']['f75']))


def _journeys(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Columns common to both journey tables: one row per journey reported (REISE), details
    for the ones picked (TRENR / RENR > 0)."""
    v = _JOURNEY_VARS[kind]
    mode_main = cat(df[v['mode']], MODE)
    same_back = flag(df[v['same_back']])
    out = {
        'hhnr': df['hhnr'],
        'person_no': df['zielpnr'],
        'journey_no': df['reise'],
        'weight': num(df['wp']),
        'is_detailed': (num(df[v['selected']]) > 0).astype(float),
        'date': parse_date(df[v['date']]),
        'purpose': cat(df[v['purpose']], JOURNEY_PURPOSE),
        'mode_main': mode_main,
        'is_passenger': flag(df[v['mode']], yes=PASSENGER, no=DRIVER, na=MODE),
        'same_mode_back': same_back,
        'mode_back': mode_main.where(same_back == 1),
        'n_companions': num(df[v['companions']]),
        'dist_km': num(df[v['dist']]),
        'regularity': cat(df[v['regularity']], REGULARITY),
        # Stops asked of day journeys only; overnight places were just listed.
        'has_places': flag(df['zwisch']) if kind == 'day' else np.nan,
        **_journey_place(df, 'orig', v['orig']),
        **_journey_place(df, 'dest', v['dest']),
    }
    return pd.DataFrame(out)


def journey_places(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stops of the day journeys (ZH1-3) and further overnight places (RS1-3), one row per
    place named."""
    out = []
    for alias, raw_prefix, place_type in (('dj', 'zh', 'stop'), ('oj', 'rs', 'overnight')):
        df = raw[alias]
        for no in (1, 2, 3):
            p = pd.DataFrame(_journey_place(df, 'place', f'{raw_prefix}{no}'))
            named = p[['place_x', 'place_municipality', 'place_in_switzerland']].notna().any(axis=1)
            out.append(p[named].assign(hhnr=df['hhnr'], person_no=df['zielpnr'],
                                       journey_no=df['reise'], place_no=no,
                                       place_type=place_type))
    return pd.concat(out, ignore_index=True)


def _journey_place(df: pd.DataFrame, prefix: str, raw_prefix: str) -> dict[str, pd.Series]:
    in_ch = _in_switzerland(df[f'{raw_prefix}_land'])
    x, y = _wgs84(num(df[f'{raw_prefix}_x']), num(df[f'{raw_prefix}_y']))
    return {
        f'{prefix}_x': x,
        f'{prefix}_y': y,
        f'{prefix}_postcode': num(df[f'{raw_prefix}_plz']),
        f'{prefix}_municipality': num(df[f'{raw_prefix}_bfs']),
        f'{prefix}_in_switzerland': in_ch,
    }


_LV03_TO_WGS84 = Transformer.from_crs(CRS_LV03, 'EPSG:4326', always_xy=True)


def _wgs84(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Journey coordinates as WGS84 longitude / latitude (`XY_CRS_BY_TABLE`). The journey
    tables give places in Switzerland in LV03 metres but places abroad in WGS84 degrees (not
    documented; seen in the data). LV03 is converted, so every place goes to LV95 the way the
    2010+ coordinates do. Anything else is emptied (logged)."""
    en = x.between(450_000, 900_000) & y.between(50_000, 350_000)
    ne = y.between(450_000, 900_000) & x.between(50_000, 350_000) & ~en
    deg = x.between(-180, 180) & y.between(-90, 90) & (y != 0)
    lon, lat = x.where(deg), y.where(deg)
    lv03 = en | ne
    if lv03.any():
        east, north = x.where(en).fillna(y.where(ne)), y.where(en).fillna(x.where(ne))
        lon[lv03], lat[lv03] = _LV03_TO_WGS84.transform(east[lv03].to_numpy(float),
                                                        north[lv03].to_numpy(float))
    other = x.notna() & y.notna() & ~(lv03 | deg)
    log = logging.warning if other.any() else logging.info
    log(f"  {x.name}/{y.name}: {int(en.sum()):,} LV03, {int(ne.sum()):,} LV03 with swapped "
        f"axes, {int(deg.sum()):,} WGS84 degrees (abroad), {int(other.sum()):,} other values "
        f"emptied")
    return lon, lat


def _passes(f26: pd.Series) -> dict[str, pd.Series]:
    """GA, half-fare and other season tickets from the combined pass question F26."""
    return {
        'pass_ga': flag(f26, yes=(2, 3), no=(1, *range(4, 18)), na=(20,)),
        'pass_halbtax': flag(f26, yes=_HALBTAX, no=range(2, 12), na=(20,)),
        'pass_season': flag(f26, yes=_SEASON, no=(1, 2, 3, 11)),
    }


def _endpoints(df: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for prefix, raw_prefix in (('orig', 's'), ('dest', 'z')):
        out |= {
            f'{prefix}_x': num(df[f'{raw_prefix}_x']),
            f'{prefix}_y': num(df[f'{raw_prefix}_y']),
            f'{prefix}_geocode_level': cat(df[f'{raw_prefix}_qal'], GEOCODE_LEVEL),
            f'{prefix}_postcode': num(df[f'{raw_prefix}_plz']),
            f'{prefix}_municipality': num(df[f'{raw_prefix}_bfs']),
            f'{prefix}_in_switzerland': _in_switzerland(df[f'{raw_prefix}_land']),
        }
    return out


def _in_switzerland(country: pd.Series) -> pd.Series:
    txt = country.astype('string').str.strip().str.lower()
    known = txt.notna() & txt.ne('').fillna(False)
    out = txt.isin(_SWITZERLAND).astype(float).where(known)
    logging.info(f"  {country.name}: {out.mean():.1%} in Switzerland; most frequent: "
                 f"{txt.value_counts().head(4).to_dict()}")
    return out
