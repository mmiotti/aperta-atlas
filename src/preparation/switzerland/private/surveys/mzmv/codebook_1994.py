"""
MTMC 1994 (Mikrozensus Verkehrsverhalten 1994) → canonical schema.

Source documentation: `MZ94DICT.pdf` (BFS, 26.7.1995), the per-table SPSS variable lists
(`hh.txt`, `zp.txt`, `wg.txt`, `et.txt`), questionnaire `EVV1994_Fragebogen.doc`.

Year specifics:
  - Up to two target persons per household (ZIELPERS).
  - Interviewed trip by trip: trips are reported units, not built from legs. Leg numbers
    restart in every trip (renumbered by the engine).
  - Missing codes are positive and per variable (mostly 8 = don't know, 9 = no answer;
    98 / 99 for two-digit codes); '.' = not asked.
  - Clock times are HHMM.
  - No coordinates: postcodes and municipalities, located by the BFS 2000 postcode points.
    No country information.
  - Distances are respondent estimates; `rdist` is corrected with the 2010 routing factors.
  - Purposes: 5 trip purposes; shopping and errands together; no 'accompany' and no 'return'
    (return trips carry the purpose of the outing; the direction is a separate question).
  - Education: one yes/no question per completed level → highest level reported.
  - Transit passes: GA and the number of weekly / monthly / annual passes were only asked of
    people with a pass other than half-fare (546); `pass_season` counts the latter.
  - Vehicles: separate car (AU) and motorbike (MO) tables, numbered by km driven; two-digit
    registration years; car engine size in bands; km driven in 1993; main use; one main user
    per vehicle (also motorbikes). No fuel. Legs do not say which car was used.
  - The reference day (ZP_ITAG) is undocumented; its range (1-395) suggests a day number from
    1 January 1994 (the survey ran into January 1995). Used as the survey date only if the
    implied weekdays agree with the reported one (ZP_WTAGF); the log reports the agreement.
"""

import logging

import pandas as pd

from preparation.switzerland.private.surveys.mzmv.engine import (
    bands, cat, flag, hhmm_to_min, num, numbers)

YEAR = 1994
RAW_DIR = 'MZ1994/ASCII(csv)/Dateien_csv'
RAW_FILES = {'hh': 'hh', 'hp': 'hp', 'zp': 'zp', 'au': 'au', 'mo': 'mo', 'wg': 'wg', 'et': 'et'}
READ_KWS = dict(sep=None, engine='python', encoding='latin-1')
XY_CRS = None

_NA = (8, 9)                    # weiss nicht, keine Angabe

SEX = {1: 'male', 2: 'female'}  # SEX

# ET06 (format E_VM): Etappen Verkehrsmittel
MODE = {
    1: 'walk',                  # zu Fuss
    2: 'rbike',                 # Velo
    3: 'motorbike_small',       # Mofa
    4: 'motorbike',             # Motorrad als Fahrer
    14: 'motorbike',            # Motorrad als Mitfahrer
    5: 'car',                   # Auto als Fahrer
    6: 'car',                   # Auto als Mitfahrer
    7: 'transit_bus',           # Bus
    8: 'transit_bus',           # Postauto
    9: 'transit_rail',          # Bahn
    17: 'transit_tram',         # Tram
    21: 'taxi',                 # Taxi
    22: 'coach',                # Reisecar
    23: 'truck',                # Lastwagen
    24: 'boat',                 # Schiff
    25: 'aerialway',            # Seilbahn
    26: 'plane',                # Flugzeug
    30: 'other',                # Ski
    31: 'micro',                # nicht motorisierte Sportgeräte
    32: 'other',                # motorisierte Sportgeräte
    40: 'other',                # Traktor
    90: 'other',                # anderes
    99: None,                   # keine Angabe
}
PASSENGER, DRIVER = (6, 14), (4, 5)

# ET03 (format E_ZWECK): Etappenzweck
LEG_PURPOSE = {
    1: 'work_commute',          # Arbeit
    2: 'education',             # Ausbildung
    3: 'errands_unspecified',   # Einkauf (Einkauf, Besorgung)
    4: 'leisure',               # Freizeit
    5: 'work_business',         # geschäftlich (geschäftlich, dienstlich)
    6: 'mode_transfer',         # Verkehrsmittel gewechselt
    9: None,                    # keine Angabe
}

# WG03 (format W_ZWECK): Wegzweck
TRIP_PURPOSE = {k: v for k, v in LEG_PURPOSE.items() if k != 6}

# WG09 (format W_ART): Wegart
DIRECTION = {
    1: 'outbound',              # Hinweg von zuhause
    2: 'outbound',              # Hinweg aus auswärtiger Unterkunft
    3: 'outbound',              # Hinweg von auswärtigem Ziel zu auswärtigem Ziel
    4: 'outbound',              # Rückweg von auswärtigem Ziel zu auswärtigem Ziel
    5: 'return',                # Rückweg nach Hause
    6: 'return',                # Rückweg in auswärtige Unterkunft
    7: 'round',                 # Rundweg
    8: None,                    # pauschal erfasste Wege (z.B. Taxifahrer)
    9: None,                    # keine Angabe
}

# HH05 (format WOHNSIT): Wohnsituation
HH_TYPE = {
    1: 'single',                # allein
    2: 'couple_no_kids',        # Lebenspartner
    3: 'family',                # Familie
    4: 'other',                 # Wohngemeinschaft
    5: 'other',                 # Untermieter
    9: None,                    # keine Angabe
}

# HP05 (format ARBEIT): Erwerbsituation (same codes as 2000 F50400, 2005 F44)
EMPLOYMENT = {
    1: 'employed',              # erwerbstätig
    2: 'employed',              # Teilzeit, eine Stelle
    3: 'employed',              # mehrere Stellen
    4: 'unemployed',            # auf Stellensuche
    5: 'unemployed',            # Job zugesichert
    6: 'other_inactive',        # nicht erwerbstätig
    7: 'homemaker',             # Hausarbeit
    8: 'in_education',          # in Ausbildung
    9: 'retired',               # RentnerIn
    10: 'other_inactive',       # andere Situation
    98: None,                   # weiss nicht
    99: None,                   # keine Angabe
}

# ZP36 (format WETTER): Wetter Stichtag
WEATHER = {
    1: 'sunny',                 # sonnig/schön
    2: 'partly_cloudy',         # leicht bewölkt
    3: 'cloudy',                # bewölkt
    4: 'foggy',                 # Nebel
    5: 'rainy',                 # Regen
    6: 'snowy',                 # Schnee
    7: 'unstable',              # veränderlich
    8: None,                    # weiss nicht
    9: None,                    # keine Angabe
}

# AU02 (format HUBRAUM): Hubraum (cars)
ENGINE_BANDS = {
    1: (0, 1350),               # bis 1350 ccm
    2: (1351, 1650),
    3: (1651, 1950),
    4: (1951, 2250),
    5: (2251, 2550),
    6: (2551, None),            # über 2550 ccm
    8: None,                    # weiss nicht
    9: None,                    # keine Angabe
}

# AU04 / MO04 (format AZWECK): Nutzungszweck
MAIN_USE = {
    1: 'commute_education',     # Arbeit/Ausbildung
    2: 'shopping_errands',      # Einkauf/Besorgungen
    3: 'leisure',               # Freizeit
    4: 'business',              # geschäftlich
    5: 'rarely_used',           # (fast) nicht
    8: None,                    # weiss nicht
    9: None,                    # keine Angabe
}

# Formats KAZWEI (2-digit year), KASECHS (odometer), KAFUENF (km), NUTZER (main user)
_YEAR_NA = (98, 99)
_ODOMETER_NA, _ODOMETER_TOP = (999998, 999999), 999997     # 999997 = more than 1 million
_KM_NA = (99998, 99999)
_NOT_A_MEMBER = (96, 97, 98, 99)    # andere Person, niemand, weiss nicht, keine Angabe

# ZP09-ZP14: "Haben Sie ... abgeschlossen?" (ja/nein), from the lowest to the highest level.
EDUCATION_QUESTIONS = (
    ('zp09', 'compulsory'),                     # Obligatorische Schule
    ('zp10', 'upper_secondary_vocational'),     # Berufslehre oder Vollzeitberufsschule
    ('zp13', 'upper_secondary_general'),        # Maturitätsschule
    ('zp11', 'tertiary_professional'),          # Höhere Berufsausbildung
    ('zp12', 'tertiary_academic'),              # Höhere Fachschule (HTL, HWV)
    ('zp14', 'tertiary_academic'),              # Hochschulstudium
)


def households(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hh = raw['hh']
    parking = num(hh['hh13'], na=_NA).mask(num(hh['hh12']) == 2, 0)
    second_homes = num(hh['hh10'], na=_NA)
    return pd.DataFrame({
        'hhnr': hh['haushalt'],
        'weight': num(hh['wm']),
        'home_postcode': num(hh['hh23']),
        'home_municipality': num(hh['gemeinde']),
        'home_canton_no': num(hh['kanton']),
        'n_people': num(hh['hh14']),
        'hh_type': cat(hh['hh05'], HH_TYPE),
        # Counts (format KATSECHS): 7 = more than six, 8 / 9 missing.
        'n_cars': num(hh['hh15'], na=_NA),
        'n_motorbikes': num(hh['hh16'], na=_NA),
        'n_mopeds': num(hh['hh17'], na=_NA),
        'n_bikes_any': num(hh['hh18'], na=_NA),
        'n_car_parking': parking,
        'has_second_home': (second_homes > 0).astype(float).where(second_homes.notna()),
    })


def people(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zp = raw['zp']
    employment_code = num(zp['hp05'])
    weekday = num(zp['zp_wtagf'])                   # Wochentag Stichtag (ZP_WTAG: last contact)
    # 546 "andere Abonnemente" (anything but half-fare, GA included); 547 GA and 548 the number
    # of weekly / monthly / annual passes were only asked if yes.
    other_passes = flag(zp['zp31'], na=_NA)
    return pd.DataFrame({
        'hhnr': zp['haushalt'],
        'person_no': zp['zielpers'],
        'weight': num(zp['wp']),
        'age': num(zp['zp01']),
        'sex': cat(zp['zp02'], SEX),
        'survey_date': _survey_date(zp['zp_itag'], weekday),
        'weekday': weekday,
        'is_mobile': flag(zp['zp39'], yes=(1, 3), na=_NA),   # 3 = whole day abroad
        'license_car': flag(zp['zp05'], na=_NA),
        'license_motorbike': flag(zp['zp06'], na=_NA),
        'pass_ga': flag(zp['zp32'], na=_NA).mask(other_passes == 0, 0),
        'pass_halbtax': flag(zp['zp30'], na=_NA),   # Halbtaxabo
        'pass_season': _season_pass(zp, other_passes),
        'education': _education(zp),
        'employment': cat(zp['hp05'], EMPLOYMENT),
        'employed_fulltime': (employment_code == 1).astype(float)
                             .where(employment_code.isin([1, 2, 3])),
        'weather': cat(zp['zp36'], WEATHER),
    })


def household_members(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hp = raw['hp']
    target = (raw['zp'].drop_duplicates(['haushalt', 'person'])
              .set_index(['haushalt', 'person'])['zielpers'])
    person_no = target.reindex(pd.MultiIndex.from_frame(hp[['haushalt', 'person']])).to_numpy()
    return pd.DataFrame({
        'hhnr': hp['haushalt'],
        'member_no': hp['person'],
        'person_no': pd.Series(person_no, index=hp.index),
        'weight': num(hp['wm']),
        'age': num(hp['hp02']),
        'sex': cat(hp['hp03'], SEX),
        'license_car': flag(hp['hp08'], na=_NA),
        'license_motorbike': flag(hp['hp09'], na=_NA),
    })


def vehicles(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    au, mo = raw['au'], raw['mo']
    cc_min, cc_max = bands(au['au02'], ENGINE_BANDS)
    moto_cc = num(mo['mo02'], na=(9998, 9999))
    cars = pd.DataFrame({
        'hhnr': au['haushalt'],
        'weight': num(au['wm']),
        'vehicle_type': 'car',
        'vehicle_no': au['fahrzeug'],
        'registration_year': num(au['au05'], na=_YEAR_NA) + 1900,
        'odometer_km': _odometer(au['au06']),
        'km_per_year': num(au['au09'], na=_KM_NA),          # 1993
        'km_per_year_abroad': num(au['au07'], na=_KM_NA),
        'engine_cc_min': cc_min,
        'engine_cc_max': cc_max,
        'main_use': cat(au['au04'], MAIN_USE),
        'main_driver_nos': numbers(au['au03'], na=_NOT_A_MEMBER),
    })
    motorbikes = pd.DataFrame({
        'hhnr': mo['haushalt'],
        'weight': num(mo['wm']),
        'vehicle_type': 'motorbike',
        'vehicle_no': mo['fahrzeug'],
        'registration_year': num(mo['mo05'], na=_YEAR_NA) + 1900,
        'odometer_km': _odometer(mo['mo06']),
        'km_per_year': num(mo['mo09'], na=_KM_NA),
        'km_per_year_abroad': num(mo['mo07'], na=_KM_NA),
        'engine_cc_min': moto_cc,
        'engine_cc_max': moto_cc,
        'main_use': cat(mo['mo04'], MAIN_USE),
        'main_driver_nos': numbers(mo['mo03'], na=_NOT_A_MEMBER),
    })
    return pd.concat([cars, motorbikes], ignore_index=True)


def _odometer(s: pd.Series) -> pd.Series:
    x = num(s, na=_ODOMETER_NA)
    return x.mask(x == _ODOMETER_TOP, 1_000_000)


def trips(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wg = raw['wg']
    return pd.DataFrame({
        'hhnr': wg['haushalt'],
        'person_no': wg['zielpers'],
        'trip_no': wg['weg'],
        'weight': num(wg['wp']),
        'dep_min': hhmm_to_min(wg['wg01']),
        'arr_min': hhmm_to_min(wg['wg02']),
        'duration_min': num(wg['wg_dauer']),
        'in_motion_min': num(wg['wg_edaue']),       # Summe der Etappendauern
        'dist_km': num(wg['w_rdist']),
        'dist_estimated_km': num(wg['wg_dist']),
        'n_legs': num(wg['wg11']),
        'purpose': cat(wg['wg03'], TRIP_PURPOSE),
        'direction': cat(wg['wg09'], DIRECTION),
        'orig_postcode': num(wg['wg04']),
        'orig_municipality': num(wg['wg_gmd']),
        'dest_postcode': num(wg['wg05']),
        'dest_municipality': num(wg['wg_zgmd']),
    })


def legs(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    et = raw['et']
    return pd.DataFrame({
        'hhnr': et['haushalt'],
        'person_no': et['zielpers'],
        'trip_no': et['weg'],
        'leg_no': et['etappe'],
        'weight': num(et['wp']),
        'dep_min': hhmm_to_min(et['et01']),
        'arr_min': hhmm_to_min(et['et02']),
        'duration_min': num(et['et_dauer']),
        'dist_km': num(et['rdist']),
        'dist_estimated_km': num(et['et_dist']),
        'mode': cat(et['et06'], MODE),
        'is_passenger': flag(et['et06'], yes=PASSENGER, no=DRIVER, na=MODE),
        'car_occupancy': num(et['et09'], na=(9, 99)),
        'purpose': cat(et['et03'], LEG_PURPOSE),
        'orig_postcode': num(et['plz']),
        'orig_municipality': num(et['gmd']),
        'dest_postcode': num(et['z_plz']),
        'dest_municipality': num(et['z_gmd']),
    })


def _season_pass(zp: pd.DataFrame, other_passes: pd.Series) -> pd.Series:
    """1 if the person holds a weekly, monthly or annual pass (548, format ABOANZ: number of
    passes, 9 = no answer); 0 if they hold no pass besides half-fare (546 = no)."""
    counts = pd.concat([num(zp[c], na=(9,)) for c in ('zp33', 'zp34', 'zp35')], axis=1)
    season = (counts.max(axis=1) > 0).astype(float).where(counts.notna().any(axis=1))
    return season.mask(other_passes == 0, 0)


def _survey_date(day_no: pd.Series, weekday: pd.Series, min_agreement: float = 0.95
                 ) -> pd.Series:
    """ZP_ITAG read as day number from 1 January 1994 (day 1 or day 0, whichever fits),
    accepted if the implied weekdays agree with ZP_WTAGF for at least `min_agreement`."""
    n = num(day_no)
    known = n.notna() & weekday.notna()
    best = None
    for first_day in (1, 0):
        dates = pd.Timestamp('1994-01-01') + pd.to_timedelta(n - first_day, unit='D')
        agreement = (dates.dt.dayofweek + 1 == weekday)[known].mean()
        if best is None or agreement > best[0]:
            best = (agreement, first_day, dates)
    agreement, first_day, dates = best
    used = agreement >= min_agreement
    logging.info(f"  zp_itag ({n.min():.0f}-{n.max():.0f}) as day number with 1 Jan 1994 = "
                 f"day {first_day}: weekday agrees with zp_wtagf for {agreement:.1%} → "
                 f"{'used' if used else 'NOT used'} as survey date")
    return dates.dt.strftime('%Y-%m-%d') if used else pd.Series(None, index=n.index, dtype=object)


def _education(zp: pd.DataFrame) -> pd.Series:
    """Highest level answered with 'yes'; 'none' if every answered question is 'no'."""
    level = pd.Series(None, index=zp.index, dtype=object)
    answered = pd.Series(False, index=zp.index)
    for col, name in EDUCATION_QUESTIONS:
        done = flag(zp[col], na=_NA)
        level = level.mask(done == 1, name)
        answered |= done.notna()
    return level.mask(level.isna() & answered, 'none')
