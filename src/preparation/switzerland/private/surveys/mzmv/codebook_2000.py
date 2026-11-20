"""
MTMC 2000 (Mikrozensus Verkehr 2000) → canonical schema.

Source documentation: `Variablenbeschreibung_DB_MZ00_inkl_korrDistanzen.pdf`,
`Datenbankstruktur.pdf`, questionnaire `Fragebogen-d.pdf`.

Year specifics:
  - Up to two target persons per household. Trips and legs carry the interview number INTNR,
    mapped to the target person no. (ZP) through the target-person table.
  - No coordinates: locations are postcodes and municipalities (street and house number only
    in large cities, unused), located by the BFS 2000 postcode points (`geo.py`). Postcode
    9998 = abroad, 9999 = border.
  - Distances are respondent estimates; `rdist` is corrected with the 2010 routing factors.
  - Shopping and errands are one purpose ('errands_unspecified').
  - Leg purpose code 9 is 'keine Angabe' in the database description but 'Rückkehr nach Hause
    bzw. auswärtige Unterkunft' in the questionnaire (<62900>). Read as 'return': it is on
    28 % of legs, matching the ~25 % return legs the trip-construction description mentions.
  - The trip table has no outbound / return flag: derived from the purpose of the trip's
    last leg (return → 'return', else 'outbound'; round trips are not identified).
  - Leisure activity only at trip level (RCODEX).
  - Transit passes come from the household-member table (HP), joined on HHNR + PERSON.
  - Vehicles: separate car (AU) and motorbike (MO) tables, numbered by km driven; car engine
    size in bands; km driven in 1999, abroad and in Switzerland; main use; no fuel, no main
    driver. Car legs say whose car and which household car (F61310), except in interviews
    from January to March 2000: F61310 is blank there (whole interviews, 22 % of car legs),
    and set from April on.
"""

import logging

import numpy as np
import pandas as pd

from preparation.switzerland.private.surveys.mzmv.engine import bands, cat, flag, num

YEAR = 2000
RAW_DIR = 'MZ2000/ASCII(csv)/Dateien'
RAW_FILES = {'hh': 'hh', 'hp': 'hp', 'zp': 'zp', 'au': 'au', 'mo': 'mo', 'wg': 'wg', 'et': 'et'}
READ_KWS = dict(sep=None, engine='python', encoding='latin-1')
XY_CRS = None

SEX = {1: 'male', 2: 'female'}

# F61300: Verkehrsmittel
MODE = {
    1: 'walk',                  # Zu Fuss
    2: 'rbike',                 # Velo
    3: 'motorbike_small',       # Mofa (Motorfahrrad)
    4: 'motorbike',             # Motorrad als Fahrer
    5: 'motorbike',             # Motorrad als Mitfahrer
    6: 'car',                   # Auto als Fahrer
    7: 'car',                   # Auto als Mitfahrer
    8: 'transit_rail',          # Bahn
    9: 'transit_bus',           # Postauto
    10: 'transit_bus',          # Bus
    11: 'transit_tram',         # Tram
    12: 'taxi',                 # Taxi
    13: 'coach',                # Reisecar
    14: 'truck',                # Lastwagen
    15: 'boat',                 # Schiff
    16: 'plane',                # Flugzeug
    17: 'aerialway',            # Zahnradbahn, Seilbahn, Standseilbahn, Sessellift, Skilift
    23: 'motorbike_small',      # Kleinmotorrad
    90: 'other',                # Anderes
    99: None,                   # Keine Angabe
}
PASSENGER, DRIVER = (5, 7), (4, 6)

# F62900 (legs) and WZWECK2 (trips): Zweck der Etappe / des Weges
PURPOSE = {
    0: 'mode_transfer',         # Umsteigen / Verkehrsmittelwechsel
    1: 'work_commute',          # Arbeit
    2: 'education',             # Ausbildung
    3: 'errands_unspecified',   # Einkauf / Besorgungen
    4: 'work_business',         # Geschäftliche Tätigkeit
    5: 'work_business',         # Dienstfahrt
    6: 'leisure',               # Freizeit
    7: 'accompany',             # Serviceweg (jemanden bringen, abholen)
    8: 'accompany',             # Begleitweg
    9: 'return',                # Rückkehr nach Hause (questionnaire; see module docstring)
}
_RETURN = 9

# RCODEX: Freizeittätigkeit (trips)
LEISURE = {
    0: None,                    # undocumented (a few dozen trips)
    1: 'visit',                 # Besuche
    2: 'outdoor_nonsport',      # nicht sportliche Aussenaktivitäten
    3: 'sport_active',          # Sport
    4: 'culture',               # Kultur, Freizeitanlagen
    5: 'gastronomy',            # Gastronomie
    6: 'unpaid_work',           # Arbeit als Freizeit
    7: 'shopping_stroll',       # Einkauf als Freizeit
    8: 'other',                 # anderes
    9: 'multiple',              # gemischt
}

# F10700: Art des Zusammenwohnens
HH_TYPE = {
    1: 'single',                # Allein
    2: 'couple_no_kids',        # LebenspartnerIn ohne Kinder
    3: 'family',                # Familie
    4: 'other',                 # Wohngemeinschaft
    5: 'other',                 # UntermieterIn
}

# EIN: Monatliches Haushaltseinkommen brutto, CHF
INCOME = {
    1: (0, 1999),               # unter 2000
    2: (2000, 4000),
    3: (4001, 6000),
    4: (6001, 8000),
    5: (8001, 10000),
    6: (10001, 12000),
    7: (12001, 14000),
    8: (14001, None),           # höher als 14000
}

# F50300: letzte abgeschlossene Ausbildung
EDUCATION = {
    0: 'none',                          # Keine Ausbildung abgeschlossen
    1: 'compulsory',                    # Obligatorische Schule
    2: 'upper_secondary_vocational',    # Berufslehre
    3: 'upper_secondary_vocational',    # Vollzeitberufsschule (z.B. Handelsschule)
    4: 'upper_secondary_general',       # Maturitätsschule, Primarlehrerausbildung
    5: 'tertiary_professional',         # Höhere Berufsausbildung
    6: 'tertiary_academic',             # Technikerschule, höhere Fachschule, Fachhochschule
    7: 'tertiary_academic',             # Universität, Hochschule
}

# F50400: berufstätig (same codes as 2005 F44)
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
    10: 'other_inactive',       # Andere Situation (Chronischkranke etc.)
}

# F60100: Wetter am Stichtag
WEATHER = {
    1: 'sunny',                 # Sonnig/schön
    2: 'partly_cloudy',         # Leicht bewölkt
    3: 'cloudy',                # Bewölkt/stark bewölkt
    4: 'foggy',                 # neblig/Nebel
    5: 'rainy',                 # Regen
    6: 'snowy',                 # Schnee
    7: 'unstable',              # Stark veränderlich/unstabil
}

# F30400: Hubraumklasse (cars; same bands as 1994)
ENGINE_BANDS = {
    1: (0, 1350),               # bis 1'350 ccm
    2: (1351, 1650),
    3: (1651, 1950),
    4: (1951, 2250),
    5: (2251, 2550),
    6: (2551, None),            # 2'551 und mehr
}

# F30600 / F31400: Hauptverwendungszweck
MAIN_USE = {
    1: 'commute_education',     # Arbeits-, Ausbildungswege
    2: 'shopping_errands',      # Einkauf, Besorgungen
    3: 'leisure',               # Freizeit
    4: 'business',              # Geschäftlich, dienstlich
    5: 'rarely_used',           # Wird (fast) nicht verwendet (Sammlerstück etc.)
}

# F61310: Art des Autos (car legs)
CAR_SOURCE = {
    **{n: 'household' for n in range(1, 8)},    # 1.-7. Auto im Haushalt
    11: 'company',              # Firmenauto / Dienstwagen
    12: 'rental',               # Mietauto
    13: 'carsharing',           # Autoteilen / Car Sharing / Mobility
    14: 'other',                # Anderes Auto
    15: None,                   # Keine Angabe
    99: None,                   # Nicht befragt
}

_ABROAD, _BORDER = 9998, 9999   # postcode codes
_ODOMETER_TOP, _KM_TOP = 999997, 99997  # "more than 1 million" / "more than 100'000" km


def households(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hh = raw['hh']
    income_min, income_max = bands(hh['ein'], INCOME)
    parking = num(hh['f11701']).mask(num(hh['f11700']) == 2, 0)
    second_homes = num(hh['f11200'])
    return pd.DataFrame({
        'hhnr': hh['hhnr'],
        'weight': num(hh['wm']),
        'home_postcode': num(hh['hhplz'], na=(9999,)),
        'home_municipality': num(hh['hhgem']),
        'home_canton_no': num(hh['kanton'], na=(99,)),
        'n_people': num(hh['hhanzper']),
        'hh_type': cat(hh['f10700'], HH_TYPE),
        'income_min_chf': income_min,
        'income_max_chf': income_max,
        'n_cars': num(hh['f30100']),
        'n_motorbikes': num(hh['f31100']),
        'n_small_motorbikes': num(hh['f32000']),
        'n_mopeds': num(hh['f32050']),
        'n_bikes_any': num(hh['f32100']),
        'n_car_parking': parking,
        'has_second_home': (second_homes > 0).astype(float).where(second_homes.notna()),
    })


def people(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zp = raw['zp']
    # Passes were asked for every household member (HP table).
    hp = (zp[['hhnr', 'person']]
          .merge(raw['hp'].drop_duplicates(['hhnr', 'person']), how='left', on=['hhnr', 'person'])
          .set_index(zp.index))
    day_of_year = num(zp['daystnr'])
    survey_date = pd.Timestamp('2000-01-01') + pd.to_timedelta(day_of_year - 1, unit='D')
    logging.info(f"  daystnr: survey days {survey_date.min():%Y-%m-%d} to "
                 f"{survey_date.max():%Y-%m-%d}")
    return pd.DataFrame({
        'hhnr': zp['hhnr'],
        'person_no': zp['zp'],
        'weight': num(zp['wp']),
        'age': num(zp['f50001']),
        'sex': cat(zp['f50002'], SEX),
        'survey_date': survey_date.dt.strftime('%Y-%m-%d'),
        'weekday': num(zp['daysttag']),
        'is_mobile': flag(zp['f60500']),
        'license_car': flag(zp['f50005']),
        'license_motorbike': flag(zp['f50006']),
        'pass_ga': flag(hp['f21200']),
        'pass_halbtax': flag(hp['f21000'], yes=(1, 2, 3), no=(4,)),   # 1-3: Halbtax variants
        'pass_season': _season_pass(hp),
        'education': cat(zp['f50300'], EDUCATION),
        'employment': cat(zp['f50400'], EMPLOYMENT),
        'employed_fulltime': flag(zp['f50400'], yes=(1,), no=(2, 3), na=EMPLOYMENT),
        'weather': cat(zp['f60100'], WEATHER),
    })


def household_members(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hp = raw['hp']
    target = raw['zp'].drop_duplicates(['hhnr', 'person']).set_index(['hhnr', 'person'])['zp']
    person_no = target.reindex(pd.MultiIndex.from_frame(hp[['hhnr', 'person']])).to_numpy()
    return pd.DataFrame({
        'hhnr': hp['hhnr'],
        'member_no': hp['person'],
        'person_no': pd.Series(person_no, index=hp.index),
        'weight': num(hp['wm']),
        'age': num(hp['persage']),
        'sex': cat(hp['perssex'], SEX),
        'license_car': flag(hp['f20800']),
        'license_motorbike': flag(hp['f20900']),
        'pass_ga': flag(hp['f21200']),
        'pass_halbtax': flag(hp['f21000'], yes=(1, 2, 3), no=(4,)),
        'pass_season': _season_pass(hp),
    })


def vehicles(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    au, mo = raw['au'], raw['mo']
    cc_min, cc_max = bands(au['f30400'], ENGINE_BANDS)
    moto_cc = num(mo['f31200'])
    cars = pd.DataFrame({
        'hhnr': au['hhnr'],
        'weight': num(au['wm']),
        'vehicle_type': 'car',
        'vehicle_no': au['auto'],
        'registration_year': num(au['f30700']),
        'registration_month': num(au['f30710']),
        'odometer_km': _top_coded(au['f30800'], _ODOMETER_TOP),
        **_km_1999(au['f30900'], au['f31000']),
        'engine_cc_min': cc_min,
        'engine_cc_max': cc_max,
        'main_use': cat(au['f30600'], MAIN_USE),
    })
    motorbikes = pd.DataFrame({
        'hhnr': mo['hhnr'],
        'weight': num(mo['wm']),
        'vehicle_type': 'motorbike',
        'vehicle_no': mo['moto'],
        'registration_year': num(mo['f31500']),
        'registration_month': num(mo['f31510']),
        'odometer_km': _top_coded(mo['f31600'], _ODOMETER_TOP),
        **_km_1999(mo['f31700'], mo['f31800']),
        'engine_cc_min': moto_cc,
        'engine_cc_max': moto_cc,
        'main_use': cat(mo['f31400'], MAIN_USE),
    })
    return pd.concat([cars, motorbikes], ignore_index=True)


def _km_1999(abroad: pd.Series, domestic: pd.Series) -> dict[str, pd.Series]:
    """Km driven in 1999, asked abroad and in Switzerland separately."""
    abroad, domestic = _top_coded(abroad, _KM_TOP), _top_coded(domestic, _KM_TOP)
    return {'km_per_year': abroad + domestic, 'km_per_year_abroad': abroad}


def _top_coded(s: pd.Series, top: int) -> pd.Series:
    """Top code ("more than ...") → its threshold, e.g. 999997 → 1,000,000."""
    x = num(s)
    return x.mask(x == top, top + 3)


def trips(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wg, et = raw['wg'], raw['et']
    last_leg_purpose = (et.assign(p=num(et['f62900']))
                        .sort_values(['intnr', 'weg', 'etappe'])
                        .drop_duplicates(['intnr', 'weg'], keep='last')
                        .set_index(['intnr', 'weg'])['p'])
    last = pd.Series(last_leg_purpose.reindex(pd.MultiIndex.from_frame(wg[['intnr', 'weg']]))
                     .to_numpy(), index=wg.index)
    direction = pd.Series(np.where(last == _RETURN, 'return', 'outbound'),
                          index=wg.index).where(last.notna())
    return pd.DataFrame({
        'hhnr': wg['hhnr'],
        'person_no': _person_no(raw, wg),
        'trip_no': wg['weg'],
        'weight': num(wg['wp']),
        'dep_min': num(wg['wvon']),
        'arr_min': num(wg['wbis']),
        'duration_min': num(wg['wdauer2']),
        'in_motion_min': num(wg['wdauer1']),
        'dist_km': num(wg['w_rdist']),
        'dist_estimated_km': num(wg['wegdist']),
        'n_legs': num(wg['etappen']),
        'purpose': cat(wg['wzweck2'], PURPOSE),
        'direction': direction,
        'leisure_activity': cat(wg['rcodex'], LEISURE),
        **_endpoint(wg, 'orig', 'w61201', 'gemvon'),
        **_endpoint(wg, 'dest', 'w61601', 'gemnach'),
    })


def legs(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    et = raw['et']
    share_return = (num(et['f62900']) == _RETURN).mean()
    logging.info(f"  f62900: code 9 (read as return home) on {share_return:.1%} of legs")
    car = num(et['f61300']).isin([6, 7])
    blank = num(et['f61310'])[car].isna()
    day = et['intnr'][car].map(raw['zp'].drop_duplicates('intnr').set_index('intnr')['daystnr'])
    early = num(day) < 92                           # before 1 April 2000
    logging.info(f"  f61310 (car type) blank on car legs: {blank[early].mean():.1%} of "
                 f"interviews January to March, {blank[~early].mean():.1%} from April")
    return pd.DataFrame({
        'hhnr': et['hhnr'],
        'person_no': _person_no(raw, et),
        'trip_no': et['weg'],
        'leg_no': et['etappe'],
        'weight': num(et['wp']),
        'dep_min': num(et['von']),
        'arr_min': num(et['bis']),
        'duration_min': num(et['edauer']),
        'dist_km': num(et['rdist']),
        'dist_estimated_km': num(et['f61500']),
        'mode': cat(et['f61300'], MODE),
        'is_passenger': flag(et['f61300'], yes=PASSENGER, no=DRIVER, na=MODE),
        'car_occupancy': num(et['f61320'], na=(9,)),       # 8 = more than 7, 9 = no answer
        'car_source': cat(et['f61310'], CAR_SOURCE),
        'car_no': num(et['f61310']).where(num(et['f61310']).between(1, 7)),
        'purpose': cat(et['f62900'], PURPOSE),
        'is_abroad': flag(et['ausland'], yes=(1,), no=(0,)),
        **_endpoint(et, 'orig', 'f61201', 'gemvon'),
        **_endpoint(et, 'dest', 'f61601', 'gemnach'),
    })


def _season_pass(hp: pd.DataFrame) -> pd.Series:
    """1 if the person holds a weekly, monthly or annual pass (F22001-F22003: number of
    passes, asked if F21100 'andere Abonnemente' = yes); 0 if F21100 = no. F21100 alone
    would also count GA holders."""
    counts = pd.concat([num(hp[c]) for c in ('f22001', 'f22002', 'f22003')], axis=1)
    season = (counts.max(axis=1) > 0).astype(float).where(counts.notna().any(axis=1))
    return season.mask(flag(hp['f21100']) == 0, 0)


def _person_no(raw: dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.Series:
    """Target person no. (ZP) for rows keyed by interview number."""
    return df['intnr'].map(raw['zp'].drop_duplicates('intnr').set_index('intnr')['zp'])


def _endpoint(df: pd.DataFrame, prefix: str, postcode_col: str,
              municipality_col: str) -> dict[str, pd.Series]:
    postcode = num(df[postcode_col])
    municipality = num(df[municipality_col])
    in_ch = pd.Series(np.select(
        [postcode == _ABROAD, postcode == _BORDER, postcode.notna() | municipality.notna()],
        [0.0, np.nan, 1.0], np.nan), index=df.index)
    return {
        f'{prefix}_postcode': postcode.where(postcode < _ABROAD),
        f'{prefix}_municipality': municipality.where(in_ch != 0),
        f'{prefix}_in_switzerland': in_ch,
    }
