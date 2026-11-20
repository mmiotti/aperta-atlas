"""
Canonical schema of the standardized MTMC (MZMV) tables, identical for every survey year.

`a_mzmv_standardize.py` writes `mzmv_<year>/standardized/<table>.csv` with exactly the columns
of `TABLES[<table>]`, in that order. A column a year did not survey is present but empty.
Categorical columns only take values from the vocabularies below (checked on every run); the
per-year codebooks map the raw BFS codes onto them.

Conventions:
  - Ids are strings: household `<year>_<HHNR>`, person `<hh_id>_<target person no.>`, trip
    `<person_id>_<trip no.>`, leg `<person_id>_<leg no.>`, journey `<person_id>_d<no.>` (day)
    or `<person_id>_o<no.>` (overnight). Until 2010 a household can have two target persons;
    from 2015 the target person no. is always 1.
  - Times are minutes after midnight of the survey day; values past 1440 run into the next day.
  - Distances are km, rounded to 1 m.
  - Columns holding only whole numbers (codes, flags, counts, times) are integers.
  - Coordinates are LV95 (EPSG:2056) metres; `<prefix>_geocode_level` says how precise.
  - Missing, not asked, don't know and no answer are all empty (NaN).
  - Flags are 1/0, empty if unknown.

Cross-year caveats that the vocabularies cannot hide are noted next to them.
"""

# Leg modes, in the BFS main-mode ranking (highest first; 2021 `wmittel1`, with Postauto
# merged into bus as from 2021). The order is used to pick a trip's ranked main mode.
MODES: tuple[str, ...] = (
    'plane',
    'transit_rail',
    'boat',
    'transit_tram',
    'transit_bus',          # incl. Postauto and school buses
    'aerialway',            # cable car, chair/ski lift, funicular, rack railway
    'coach',
    'car',
    'truck',
    'taxi',                 # incl. ride-hailing (2021)
    'motorbike',
    'motorbike_small',      # moped (Mofa) and small motorbike (Kleinmotorrad)
    'ebike45',              # 2015 on
    'ebike25',              # 2015 on
    'anybike',              # bike, e-bike not told apart (2010)
    'rbike',                # bike without assist (2015 on; also all bikes up to 2005)
    'walk',
    'micro',                # scooters, skates etc. (Fahrzeugähnliche Geräte)
    'other',
)

# Activity at the destination (legs) and trip purpose (trips). The BFS assigns return trips
# home the purpose of the longest activity of the outing, so 'return' is rare in trips.
PURPOSES: tuple[str, ...] = (
    'mode_transfer',        # change of mode, parking the car (legs only)
    'work_commute',         # working at the destination
    'education',
    'errands_groceries',    # shopping (2005 on)
    'errands_services',     # errands, using services (2005 on)
    'errands_unspecified',  # shopping or errands, not told apart (1994, 2000)
    'work_business',        # business activity or business trip
    'leisure',
    'accompany',            # accompanying or picking up others (2000 on)
    'return',               # return home or to the accommodation (legs, 2000 on)
    'border_crossing',      # leg ends at the Swiss border (2005 on)
    'other',
)

# Leisure activity (purpose 'leisure'). Hiking and cycling are split off only from 2010.
LEISURE_ACTIVITIES: tuple[str, ...] = (
    'visit',
    'gastronomy',
    'sport_active',
    'hiking',
    'cycling',
    'sport_passive',
    'outdoor_nonsport',     # walks and other outdoor activities
    'wellness',             # medical, wellness, fitness
    'culture',              # culture, leisure facilities
    'unpaid_work',
    'club',
    'excursion',            # excursion, holiday
    'religion',
    'home_activity_away',   # home-type leisure elsewhere (e.g. at neighbours)
    'picnic',               # eating out without a restaurant
    'shopping_stroll',
    'round_trip',           # ride for its own sake
    'accompany',            # accompanying as leisure (2005)
    'multiple',
    'other',
)

TRIP_DIRECTIONS: tuple[str, ...] = (
    'outbound',             # destination is not home
    'return',               # to home or the accommodation
    'round',                # from home to home
)

SEXES: tuple[str, ...] = ('male', 'female')

# Highest completed education. 'tertiary_academic' includes the colleges of higher education
# (HF, HTL, HWV) because 2000 does not tell them apart from universities of applied sciences.
EDUCATION: tuple[str, ...] = (
    'none',
    'compulsory',
    'upper_secondary_vocational',
    'upper_secondary_general',     # incl. vocational baccalaureate and teacher seminar
    'tertiary_professional',       # federal professional exams
    'tertiary_academic',
)

# Up to 2005 the respondent's self-assessed main activity; from 2010 the BFS labour-force
# status (ILO). Apprentices count as 'in_education' in all years.
EMPLOYMENT: tuple[str, ...] = (
    'employed',
    'unemployed',
    'in_education',
    'retired',
    'homemaker',
    'other_inactive',
)

# Weather on the survey day (first mention).
WEATHER: tuple[str, ...] = (
    'sunny', 'partly_cloudy', 'cloudy', 'foggy', 'rainy', 'snowy', 'unstable', 'hot', 'cold',
)

HH_TYPES: tuple[str, ...] = (
    'single',
    'couple_no_kids',
    'couple_kids',
    'single_parent',
    'family',               # couple or single parent with kids, not told apart (1994, 2000)
    'other',                # flat shares, subtenants, other non-family households
)

VEHICLE_TYPES: tuple[str, ...] = ('car', 'motorbike')

# Fuel / powertrain (cars, 2005 on).
FUELS: tuple[str, ...] = (
    'petrol',
    'diesel',
    'hybrid',               # not plug-in, or plug-in status unknown (no registry data)
    'plugin_hybrid',        # incl. range extenders; see the codebooks for how identified
    'electric',
    'gas',                  # CNG / LPG
    'ethanol',              # E85
    'alternative',          # hybrid, E85 or gas, not told apart (2010)
    'other',
)

# Main use of a vehicle (1994, 2000).
MAIN_USES: tuple[str, ...] = (
    'commute_education', 'shopping_errands', 'leisure', 'business', 'rarely_used',
)

# Vehicle registry (MOFIS) attributes, 2010 on.
TRANSMISSIONS: tuple[str, ...] = (
    'automatic', 'manual', 'hydrostatic', 'continuously_variable', 'automated_manual',
)
DRIVES: tuple[str, ...] = ('front', 'rear', 'all_wheel', 'tracks', 'off_road')

# Which car a car leg used (2000 on).
CAR_SOURCES: tuple[str, ...] = ('household', 'company', 'rental', 'carsharing', 'other')

# Extra question module a target person was assigned (2005 on; about a third each for the
# two journey modules).
EXTRA_MODULES: tuple[str, ...] = (
    'day_journeys',         # module 1a
    'overnight_journeys',   # module 1b
    'slow_modes',           # module 2: walking and cycling (2021: and work situation)
    'opinions',             # module 3: transport-policy opinions
)

# Main purpose of a day or overnight journey (2005 on). 2005 does not split sport into
# active, hiking, cycling and passive ('sport'), and has no education, shopping or private
# accompanying; 'unpaid_work' and 'multiple' are 2005 only.
JOURNEY_PURPOSES: tuple[str, ...] = (
    'education',            # incl. school trips and camps
    'shopping',
    'medical',
    'business',
    'accompany_business',
    'accompany_private',
    'visit',
    'gastronomy',
    'sport',                # any sport (2005)
    'sport_active',
    'hiking',
    'cycling',
    'sport_passive',
    'outdoor_nonsport',     # walks, enjoying nature
    'culture',              # culture, leisure facilities
    'excursion_holiday',
    'religion',
    'round_trip',           # ride for its own sake
    'unpaid_work',
    'multiple',
    'other',
)

# How often the person travels to the journey's destination.
REGULARITIES: tuple[str, ...] = (
    'weekly',               # at least once a week
    'monthly',
    'yearly',
    'less_often',
    'first_time',
)

# Places of a journey besides start and main destination (`journey_places`).
PLACE_TYPES: tuple[str, ...] = (
    'stop',                 # day journey: another place with an activity of over 30 minutes
    'overnight',            # overnight journey: a further overnight place
)

GEOCODE_LEVELS: tuple[str, ...] = (
    'address',              # house number, building, POI or stop
    'street',
    'postcode',             # BFS postcode point (1994, 2000)
    'municipality',
    'region',               # NUTS-3 region abroad
    'country',
)

# Column → vocabulary, checked on every table that has the column; `<table>.<column>`
# overrides it for one table.
CATEGORIES: dict[str, tuple[str, ...]] = {
    'day_journeys.purpose': JOURNEY_PURPOSES,
    'overnight_journeys.purpose': JOURNEY_PURPOSES,
    'mode_main': MODES,
    'mode_back': MODES,
    'regularity': REGULARITIES,
    'extra_module': EXTRA_MODULES,
    'place_type': PLACE_TYPES,
    'sex': SEXES,
    'education': EDUCATION,
    'employment': EMPLOYMENT,
    'weather': WEATHER,
    'hh_type': HH_TYPES,
    'mode': MODES,
    'mode_main_ranked': MODES,
    'mode_main_longest': MODES,
    'purpose': PURPOSES,
    'leisure_activity': LEISURE_ACTIVITIES,
    'direction': TRIP_DIRECTIONS,
    'home_geocode_level': GEOCODE_LEVELS,
    'orig_geocode_level': GEOCODE_LEVELS,
    'dest_geocode_level': GEOCODE_LEVELS,
    'vehicle_type': VEHICLE_TYPES,
    'fuel': FUELS,
    'main_use': MAIN_USES,
    'transmission': TRANSMISSIONS,
    'drive': DRIVES,
    'car_source': CAR_SOURCES,
}

_ENDPOINT_COLUMNS = {
    '{p}_x': 'Easting (LV95).',
    '{p}_y': 'Northing (LV95).',
    '{p}_geocode_level': 'Precision of the coordinates.',
    '{p}_postcode': 'Postcode (Swiss postcodes only).',
    '{p}_municipality': 'BFS municipality number, as of the survey year.',
    '{p}_in_switzerland': '1 if in Switzerland.',
}


def _endpoint(prefix: str) -> dict[str, str]:
    return {k.format(p=prefix): v for k, v in _ENDPOINT_COLUMNS.items()}


def _journey_endpoints(*prefixes: str) -> dict[str, str]:
    """Places of a journey: a place name the BFS geocoded, so no better than municipality
    precision and no geocode level."""
    return {k: v for p in prefixes for k, v in _endpoint(p).items()
            if not k.endswith('_geocode_level')}


def _journey(kind: str) -> dict[str, str]:
    """Columns shared by the two journey tables, `kind` 'day' or 'overnight'."""
    return {
        'journey_id': f'Journey id, `<person_id>_{kind[0]}<journey no.>`.',
        'person_id': 'Person id.',
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'journey_no': 'Number among the journeys the person reported.',
        'weight': 'Person weight (WP), for counts over all journeys.',
        'is_detailed': '1 if the journey was picked (at random, up to 3 per person) for the '
                       'detailed questions; otherwise only the date (and, where asked of all '
                       'journeys, regularity and nights) is known.',
        'weight_detailed': 'Person weight × journeys reported / journeys detailed, for analyses '
                           'of the detailed journeys.',
    }


_JOURNEY_DETAILS = {
    'purpose': 'Main purpose.',
    'mode_main': 'Main mode: the one covering the longest distance (2010 on: of the way out).',
    'is_passenger': '1 if passenger, 0 if driver (main mode car or motorbike).',
    'same_mode_back': '1 if the same main mode was used on the way back.',
    'mode_back': 'Main mode of the way back (2005: only where the same as the way out).',
    'n_companions': 'Number of people accompanying the respondent (2005-2015).',
    'dist_km': "Respondent's estimate of the whole journey (way out and back plus travel "
               'at the destination); 2015 on with BFS imputations where not known.',
    'dist_domestic_km': 'Of which in Switzerland (2010 on).',
    'regularity': 'How often the person travels to the destination.',
    'has_places': '1 if the person named stops (day) or further overnight places (overnight); '
                  'see `journey_places` (not asked for 2005 overnight journeys).',
}


# Table → column → description. The first column is the table's id.
TABLES: dict[str, dict[str, str]] = {
    'households': {
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'weight': 'Household weight (WM).',
        **{k: v for k, v in _endpoint('home').items() if not k.endswith('_in_switzerland')},
        'home_canton_no': 'BFS canton number (1-26).',
        'n_people': 'Household size.',
        'hh_type': 'Household type.',
        'income_min_chf': 'Gross monthly household income, lower bound of the band (2000 on).',
        'income_max_chf': 'Upper bound of the income band; empty for the top band.',
        'n_cars': 'Cars.',
        'n_motorbikes': 'Motorbikes.',
        'n_small_motorbikes': 'Small motorbikes (2000 on; before: in motorbikes or mopeds).',
        'n_mopeds': 'Mopeds (Mofas).',
        'n_bikes_any': 'Bikes in working order, incl. e-bikes.',
        'n_ebikes_slow': 'E-bikes up to 25 km/h (2015 on).',
        'n_ebikes_fast': 'E-bikes up to 45 km/h (2015 on).',
        'n_car_parking': 'Car parking spaces at home.',
        'has_second_home': '1 if the household has a second home.',
    },
    'household_members': {
        'member_id': 'Household-member id, `<hh_id>_m<member no.>`.',
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'weight': 'Household weight (WM).',
        'member_no': 'Member number within the household.',
        'person_id': 'Person id if this member is a target person (2015 on: inferred).',
        'age': 'Age in years.',
        'sex': 'Sex.',
        'license_car': '1 if the member holds a car driving licence (0 under 18).',
        'license_motorbike': '1 if the member holds a motorbike licence (0 under 18).',
        'pass_ga': '1 if the member holds a GA travelcard (2000, 2005).',
        'pass_halbtax': '1 if the member holds a half-fare card (2000, 2005).',
        'pass_season': '1 if the member holds another season ticket (2000, 2005).',
    },
    'people': {
        'person_id': 'Person id (target persons only).',
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'weight': 'Person weight (WP).',
        'age': 'Age in years.',
        'sex': 'Sex.',
        'survey_date': 'Survey day, YYYY-MM-DD (1994: derived, see its codebook).',
        'weekday': 'Weekday of the survey day, 1 = Monday ... 7 = Sunday.',
        'is_mobile': '1 if the person left home on the survey day.',
        'license_car': '1 if the person holds a car driving licence (0 under 18).',
        'license_motorbike': '1 if the person holds a motorbike licence (0 under 18).',
        'pass_ga': '1 if the person holds a GA travelcard.',
        'pass_halbtax': '1 if the person holds a half-fare card.',
        'pass_regional': '1 if the person holds a regional zone pass (Verbund-Abo; 2010 on).',
        'pass_season': '1 if the person holds any season ticket other than GA or half-fare.',
        'education': 'Highest completed education; empty under 15.',
        'employment': 'Employment status; empty under 15.',
        'employed_fulltime': '1 if employed full-time, 0 if part-time (employed only).',
        'weather': 'Weather on the survey day.',
        'extra_module': 'Extra question module assigned (2005 on).',
        'n_day_journeys': 'Day journeys in the 14 days before the interview (module '
                          'day_journeys).',
        'n_overnight_journeys': 'Journeys with an overnight stay in the last 4 months (module '
                                'overnight_journeys; 2021: ended in the last 4 months).',
    },
    'vehicles': {
        'vehicle_id': 'Vehicle id, `<hh_id>_<vehicle type><vehicle no.>`, e.g. `2015_123_car2`.',
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'weight': 'Household weight (WM).',
        'vehicle_type': 'Car or motorbike.',
        'vehicle_no': 'Number among the household\'s vehicles of this type (the "2nd car").',
        'registration_year': 'Year of first registration.',
        'registration_month': 'Month of first registration (partly asked).',
        'odometer_km': 'Odometer reading (km).',
        'km_per_year': 'Km driven in a year: the calendar year before the survey (1994, 2000) '
                       'or the last 12 months (2005 on).',
        'km_per_year_abroad': 'Of which abroad.',
        'engine_cc_min': 'Engine displacement (ccm), lower bound (exact value: min = max).',
        'engine_cc_max': 'Engine displacement, upper bound; empty for the top band.',
        'fuel': 'Fuel / powertrain (cars, 2005 on).',
        'main_use': 'Main use (1994, 2000).',
        'main_driver_ids': 'Member ids of the main driver(s), `;`-separated (cars; 1994 also '
                           'motorbikes; not 2000).',
        'in_registry': '1 if matched in the vehicle registry (MOFIS; 2010 on).',
        'weight_total_kg': 'Gross vehicle weight (registry, 2010 on).',
        'weight_empty_kg': 'Empty weight (registry, 2015 on).',
        'payload_kg': 'Permitted payload (registry, 2015 on).',
        'power_kw': 'Engine power (registry, 2015 on).',
        'cylinders': 'Number of cylinders (registry, 2010 on).',
        'transmission': 'Gearbox (registry, 2010 on).',
        'drive': 'Driven wheels (registry, 2015 on).',
        'co2_g_km': 'CO2 emissions, type approval (NEDC), g/km (registry, 2015 on).',
        'fuel_l_100km': 'Fuel consumption, type approval (NEDC), l/100 km (registry, 2021).',
        'electricity_kwh_100km': 'Electricity consumption, type approval (NEDC), kWh/100 km '
                                 '(registry, 2021).',
        'energy_label': 'Energy-efficiency label class, raw code (registry, 2010 on).',
        'emission_code': 'Emission code, raw (registry, 2010 on).',
        'type_approval_no': 'Type-approval number, raw (registry, 2015 on).',
    },
    'trips': {
        'trip_id': 'Trip id.',
        'person_id': 'Person id.',
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'trip_no': 'Trip number of the survey day.',
        'weight': 'Person weight (WP).',
        'dep_min': 'Departure, minutes after midnight.',
        'arr_min': 'Arrival, minutes after midnight.',
        'duration_min': 'Trip duration incl. waiting and transfer times (min).',
        'in_motion_min': 'Sum of the leg durations, i.e. without waits and transfers (min).',
        'dist_km': 'Distance (km): routed for motorised legs from 2010, BFS-corrected estimate '
                   'before; walk/bike legs: the estimate up to 2010, mostly in 2015, mostly '
                   'not in 2021 (README).',
        'dist_estimated_km': "Respondent's distance estimate (km).",
        'n_legs': 'Number of legs.',
        'has_pseudo_leg': '1 if a leg is a pseudo leg (not described in detail; 2005 on).',
        'purpose': 'Trip purpose.',
        'direction': 'Outbound, return or round trip.',
        'leisure_activity': 'Leisure activity (leisure trips; 2000 on).',
        'mode_main_ranked': 'Main mode by the BFS ranking of the leg modes.',
        'mode_main_longest': 'Mode of the longest leg (by `dist_km`).',
        **_endpoint('orig'),
        **_endpoint('dest'),
    },
    'legs': {
        'leg_id': 'Leg id.',
        'trip_id': 'Trip id.',
        'person_id': 'Person id.',
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'trip_no': 'Trip number of the survey day.',
        'leg_no': 'Leg number of the survey day.',
        'weight': 'Person weight (WP).',
        'dep_min': 'Departure, minutes after midnight.',
        'arr_min': 'Arrival, minutes after midnight.',
        'duration_min': 'Leg duration (min).',
        'dist_km': 'Distance (km), see trips.',
        'dist_estimated_km': "Respondent's distance estimate (km).",
        'mode': 'Mode.',
        'is_passenger': '1 if passenger, 0 if driver (cars and motorbikes).',
        'car_occupancy': 'Persons in the car, incl. the respondent.',
        'car_source': 'Whose car a car leg used (2000 on).',
        'vehicle_id': 'The household car used (car_source household; 2000 on).',
        'purpose': 'Activity at the destination.',
        'leisure_activity': 'Leisure activity (purpose leisure; 2005 on).',
        'is_abroad': '1 if the leg is outside Switzerland (2000 on).',
        'is_pseudo': '1 for pseudo legs (not described in detail; 2005 on).',
        **_endpoint('orig'),
        **_endpoint('dest'),
    },
    'day_journeys': {
        **_journey('day'),
        'date': 'Day of the journey, YYYY-MM-DD.',
        'duration_h': 'Duration (hours).',
        **_JOURNEY_DETAILS,
        **_journey_endpoints('orig', 'dest'),
    },
    'overnight_journeys': {
        **_journey('overnight'),
        'date': 'Start of the journey, YYYY-MM-DD.',
        'nights': 'Nights away.',
        **_JOURNEY_DETAILS,
        **_journey_endpoints('orig', 'dest'),
    },
    'journey_places': {
        'place_id': 'Place id, `<journey_id>_<place no.>`.',
        'journey_id': 'Day or overnight journey id.',
        'person_id': 'Person id.',
        'hh_id': 'Household id.',
        'year': 'Survey year.',
        'place_no': 'Number of the place within the journey (1-3, in the order named).',
        'place_type': 'Stop of a day journey or further overnight place.',
        'weight_detailed': "The journey's `weight_detailed`.",
        **_journey_endpoints('place'),
    },
}
