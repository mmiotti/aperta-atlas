# Standardized MTMC tables

The Swiss Mobility and Transport Microcensus (MTMC; German *Mikrozensus
Mobilität und Verkehr*, MZMV) is run by the Federal Statistical Office (BFS)
and the Federal Office for Spatial Development (ARE) about every five years.
`../a_mzmv_standardize.py` turns the raw deliveries of 1994, 2000, 2005, 2010,
2015 and 2021 into nine tables with the same columns and the same English codes
for every year:

| Table | One row per | Id |
|---|---|---|
| `households.csv` | household | `hh_id` |
| `household_members.csv` | person living in the household | `member_id` |
| `people.csv` | target person (the household member interviewed about their day) | `person_id` |
| `vehicles.csv` | car or motorbike of the household | `vehicle_id` |
| `trips.csv` | trip (*Weg*) | `trip_id` |
| `legs.csv` | leg (*Etappe*): part of a trip on one mode | `leg_id` |
| `day_journeys.csv` | day journey (*Tagesreise*) in the 14 days before the interview | `journey_id` |
| `overnight_journeys.csv` | journey with an overnight stay (*Reise mit Übernachtung*) in the last 4 months | `journey_id` |
| `journey_places.csv` | stop of a day journey, or further overnight place of an overnight journey | `place_id` |

> **Journeys and trips overlap.** `trips` and `legs` cover all travel on the
> sampled reference day, so they already contain part of the journeys. A day
> journey made on that day is there as ordinary trips. So is the way home from
> an overnight journey that ends that day, but usually not the way out (people
> are rarely reachable the day after they leave). Don't add the journey tables
> to the trips. See [Journeys](#journeys) for how they relate.

They are written to
`<DATA_DIR_PRIVATE>/preparation/switzerland/surveys/mzmv_<year>/standardized/`.
The raw data is BFS-restricted.

Two more scripts add tables for analyses by place:

| Table | One row per | Id | Script |
|---|---|---|---|
| `household_context.csv` | household: the [areas](#areas) of its home and the [density](#densities) around it | `hh_id` | `../b_mzmv_context.py` |
| `trip_orig_context.csv` | trip: the same for its origin | `trip_id` | `../b_mzmv_context.py` |
| `municipality_density.csv` | municipality and LUMOS year (1990–2025): [density](#densities) | `municipality`, `year` | `../b_mzmv_area_densities.py` |
| `postcode_density.csv` | postcode and LUMOS year: [density](#densities) | `postcode`, `year` | `../b_mzmv_area_densities.py` |

The context tables sit next to the standardized tables of their year. The
density tables cover all years and are in
`<DATA_DIR_PRIVATE>/preparation/switzerland/surveys/mzmv_areas/`.

A column that a year did not survey is present but empty; a table it did not
survey (the journey tables in 1994 and 2000) has the header only. Missing, not
asked, don't know and no answer are all empty. Flags are 1/0. Distances are
rounded to 1 m. Columns that hold only whole numbers (codes, flags, counts,
times, coordinates) are written without decimals.

Each run logs, per year, the checks this README quotes (e.g. how ids link up,
how distances relate, which raw codes were not documented). Read the log after
a year's first run.

## Where the mapping lives

- `schema.py`: every column with a one-line description, and the allowed
  categories. Every run checks the output against it.
- `codebook_<year>.py`: how each column is read from that year's raw
  variables, with the German source labels as comments, plus a list of that
  year's peculiarities at the top.
- `engine.py`, `geo.py`: the steps shared by all years (reading, ids,
  coordinates, main modes, checks).

A raw code that a codebook does not list becomes empty and is logged as a
warning. That is where documentation and data disagree; fix it in the
codebook.

## The survey years at a glance

| | 1994 | 2000 | 2005 | 2010 | 2015 | 2021 |
|---|---|---|---|---|---|---|
| Target persons per household | 1–2 | 1–2 | 1–2 | 1–2 | 1 | 1 |
| How trips were recorded | reported as trips | built from legs (rule A) | rule A | rule B | rule C | rule C |
| Locations in the raw data | postcode, municipality | postcode, municipality | coordinates (LV03; journeys abroad WGS84) | coordinates (WGS84) | coordinates (WGS84) | coordinates (WGS84) |
| `dist_km` for car, public transport | corrected estimate | corrected estimate | corrected estimate | routed | routed and checked | routed and checked |
| `dist_km` for walk, bike | estimate | estimate | estimate | estimate | mostly estimate (¾ of legs) | mostly not the estimate |
| Route geometries (raw) | — | — | — | ✓ | ✓ | ✓ |
| Which car a car leg used | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Vehicle registry data | — | — | — | ✓ | ✓ | ✓ |
| Target person among the household members | stated | stated | stated | stated | inferred | inferred |
| Bike mode | `rbike` | `rbike` | `rbike` | `anybike` | `rbike` / `ebike25` / `ebike45` | `rbike` / `ebike25` / `ebike45` |
| Employment concept | own main activity | own main activity | own main activity | labour-force status | labour-force status | labour-force status |
| Leisure activity detail | — | trips only, 9 kinds | 17 kinds | 17 kinds | 17 kinds | 17 kinds |
| Day and overnight journeys | — | — | ✓ | ✓ | ✓ | ✓ |

Trip rules, as applied by the BFS when it cut each person's legs into trips:

- **A**: the next leg belongs to the same trip if it has the same purpose, that
  purpose is not "change of mode", and it starts within an hour.
- **B**: as A, but a leg with the purpose "change of mode" also continues the
  trip.
- **C**: as B, and a change of leisure activity also starts a new trip.

## Which column has data in which year

✓ = available, ≈ = available with a caveat (see the notes below), — = not in
the data (not surveyed, or not kept; see the notes).

| Column | 1994 | 2000 | 2005 | 2010 | 2015 | 2021 |
|---|---|---|---|---|---|---|
| **households** | | | | | | |
| home location, postcode, municipality, canton | ≈ | ≈ | ✓ | ✓ | ✓ | ✓ |
| `n_people`, `n_cars`, `n_motorbikes`, `n_mopeds`, `n_bikes_any`, `n_car_parking`, `has_second_home` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hh_type` | ≈ | ≈ | ✓ | ✓ | ✓ | ✓ |
| `income_min_chf`, `income_max_chf` | — | ≈ | ✓ | ✓ | ✓ | ✓ |
| `n_small_motorbikes` | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `n_ebikes_slow`, `n_ebikes_fast` | — | — | — | — | ✓ | ✓ |
| **household_members** | | | | | | |
| `age`, `sex`, licences | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `person_id` (which member is a target person) | ✓ | ✓ | ✓ | ✓ | ≈ | ≈ |
| `pass_ga`, `pass_halbtax`, `pass_season` | — | ≈ | ≈ | — | — | — |
| **people** | | | | | | |
| `age`, `sex`, `weekday`, `is_mobile`, licences | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `survey_date` | ≈ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pass_ga`, `pass_halbtax`, `pass_season` | ≈ | ≈ | ≈ | ✓ | ✓ | ✓ |
| `pass_regional` | — | — | — | ✓ | ✓ | ✓ |
| `education` | ≈ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `employment`, `employed_fulltime` | ≈ | ≈ | ≈ | ✓ | ✓ | ✓ |
| `weather` | ✓ | ✓ | ≈ | ✓ | ✓ | ✓ |
| `extra_module`, `n_day_journeys`, `n_overnight_journeys` | — | — | ✓ | ✓ | ✓ | ✓ |
| **vehicles** | | | | | | |
| registration year, odometer, km per year | ≈ | ≈ | ✓ | ✓ | ✓ | ✓ |
| engine size | ≈ | ≈ | ✓ | ✓ | ✓ | ≈ |
| `fuel` | — | — | ≈ | ≈ | ✓ | ✓ |
| `main_use` | ✓ | ✓ | — | — | — | — |
| `main_driver_ids` | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| registry: gross weight, cylinders, gearbox, energy label, emission code | — | — | — | ✓ | ✓ | ✓ |
| registry: empty weight, payload, power, drive, `co2_g_km`, type approval | — | — | — | — | ✓ | ✓ |
| registry: `fuel_l_100km`, `electricity_kwh_100km` | — | — | — | — | — | ✓ |
| **trips** | | | | | | |
| times, `dist_km`, `n_legs`, `purpose`, main modes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dist_estimated_km` | ✓ | ✓ | ✓ | ✓ | — | — |
| `direction` | ✓ | ≈ | ✓ | ✓ | ✓ | ✓ |
| `leisure_activity` | — | ≈ | ≈ | ✓ | ✓ | ✓ |
| `has_pseudo_leg` | — | — | ✓ | ✓ | ✓ | ✓ |
| origin, destination | ≈ | ≈ | ✓ | ✓ | ✓ | ✓ |
| **legs** | | | | | | |
| times, `dist_km`, `mode`, `is_passenger`, `car_occupancy`, `purpose` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dist_estimated_km` | ✓ | ✓ | ✓ | ✓ | ≈ | ≈ |
| `leisure_activity` | — | — | ≈ | ✓ | ✓ | ✓ |
| `is_abroad` | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `is_pseudo` | — | — | ✓ | ✓ | ✓ | ✓ |
| `car_source`, `vehicle_id` | — | ≈ | ✓ | ✓ | ✓ | ✓ |
| origin, destination | ≈ | ≈ | ✓ | ✓ | ✓ | ✓ |
| **day_journeys, overnight_journeys** | | | | | | |
| `date`, `duration_h` / `nights`, `mode_main`, `same_mode_back`, `dist_km`, `regularity`, start, destination | — | — | ✓ | ✓ | ✓ | ✓ |
| `purpose` | — | — | ≈ | ✓ | ✓ | ✓ |
| `mode_back` | — | — | ≈ | ✓ | ✓ | ✓ |
| `n_companions` | — | — | ✓ | ✓ | ✓ | — |
| `dist_domestic_km` | — | — | — | ✓ | ✓ | ✓ |
| `has_places` | — | — | ≈ | ✓ | ✓ | ✓ |
| **journey_places** | — | — | ✓ | ✓ | ✓ | ✓ |

## Notes on the columns

### Ids and weights

- `hh_id` is `<year>_<HHNR>`. `person_id` adds the target-person number: 1 or
  2 up to 2010 (households with four or more members aged 6+ had two target
  persons), always 1 from 2015. `trip_id` and `leg_id` add the trip and leg
  number of that person's day. `journey_id` adds `d` (day) or `o` (overnight)
  and the journey number, e.g. `2015_123_1_d2`; `place_id` adds the place
  number to that. Member and vehicle ids: see their sections.
- `weight`: the household weight (`WM`) in `households`, `household_members`
  and `vehicles`; the person weight (`WP`) in `people`, `trips`, `legs` and the
  journey tables, as the BFS prescribes. The journey tables add
  `weight_detailed` (see Journeys); `journey_places` has only that, copied
  from its journey.
- **Weight anything you compare across years.** The sample design changed in
  2015. Up to 2010 the survey drew households, then one or two target persons
  in each. From 2015 it draws a person from the population register and
  surveys their household, so larger households are more likely to be in the
  sample. The household weight undoes this. Cars per household, for example,
  go from 1.17 (2010) to 1.39 (2015) unweighted, but from 1.17 to 1.16
  weighted; the weighted 2010 value is the BFS's published 1.173.
- **Household size stays a break even weighted.** The 2000–2010 household
  weights do not adjust for household size, and those samples have too few
  one-person households: 26–29 % weighted, against 32 % in 1994 and 34–36 %
  from 2015 (a third or more of Swiss households have been one-person
  households since 1990). Mean household size is higher accordingly
  (2.33–2.45, against 2.34 in 1994 and 2.17–2.21 from 2015). Since one-person
  households own fewer cars, household car ownership in 2000–2010 is likely
  somewhat overstated.
- **The person weights share this, but less.** Weighted with `WP`, 15–16 % of
  the target persons live alone in 2000–2010, against 17 % in 1994 and
  18–19 % from 2015 (the household weights imply 11.5–13 % for the same
  ages). Living alone has become steadily more common, so the 2000–2010
  person-weighted shares are probably 1.5–3 points too low, roughly a tenth
  too few people living alone. Overall per-person figures (e.g. distance per
  day) should barely move; analyses by household size or type are affected
  more.

> **Open: no household-size correction yet for 2000–2010.**
>
> *How big the problem is.* With the BFS weights, 2000–2010 has about a
> fifth to a quarter too few one-person households, and about a tenth too few
> people living alone. Overall per-person averages (trips or distance per
> day) should move by half a percent or less. What it affects:
>
> - household-level figures and their trend across 2010 → 2015 (household
>   size, car ownership: roughly 4 % too high in 2010, a rough estimate);
> - any analysis by household size or household type, at household or person
>   level.
>
> *The plan, if needed.* Add an adjusted weight next to the BFS weight, which
> stays as `weight`. The household and person weights would be rescaled so
> that, in every survey year, the distribution by household size (1, 2, 3, 4,
> 5+) matches the official statistics: the 1990 and 2000 censuses, the 2010
> structural survey and the population register from 2015, interpolated for
> 1994 and 2005. The regional distribution would be kept as the BFS weights
> have it (raking on both). It would apply to all years, so the column means
> the same everywhere; in 1994 and 2015/2021 its factors would be close to 1.
> The case is stronger for household figures than for person-level ones.

### Times

- `dep_min`, `arr_min`: minutes after midnight of the survey day; values above
  1440 run into the next day.
- Times are self-reported, mostly to the nearest 5 minutes.
- **`duration_min` vs `in_motion_min`** (trips): `duration_min` runs from the
  first departure to the last arrival, so it includes waiting and transfer
  times. `in_motion_min` is the sum of the leg durations, without them. Legs
  only have their own moving time. Use `duration_min` for door-to-door trip
  time, `in_motion_min` or the legs for time spent on a mode.

### Distances

- **`dist_km` vs `dist_estimated_km`**: `dist_km` is the distance the BFS uses
  in its own statistics (raw `rdist`; trips sum their legs).
  `dist_estimated_km` is the respondent's own estimate.
  - From 2010, `dist_km` is routed on the road or rail network for car and
    public-transport legs.
  - Before 2010 there was no routing. The BFS scaled the estimates of car,
    public-transport and other legs with correction factors derived from the
    2010 routing, by mode group and distance band. This makes the years
    comparable on average, but a single pre-2010 leg is still a guess.
  - Walk and bike legs keep the estimate up to 2010, so the two columns are
    identical there.
  - 2015 and 2021 call `dist_km` only a "checked" distance, without saying
    how walk and bike legs got theirs. The data shows a change (log line "Leg
    dist_km vs the respondent's estimate"). In 2015, 74 % of walk and 77 % of
    bike legs still have the estimate; the rest differ, often by a lot (90th
    percentile of the ratio 2.4 for walking), as if implausible estimates were
    replaced. In 2021 only 22 % of walk and 9 % of bike legs equal the
    estimate, so most were derived otherwise, presumably routed. Walk and bike
    distances are therefore not measured the same way across years; compare
    them with care.
  - In 2015 and 2021 the estimate was not asked when routing worked, so
    `dist_estimated_km` is patchy for legs and missing for trips.

  **Which to use**: `dist_km` for analysis, since it is the series the BFS
  publishes (with the method changes above). `dist_estimated_km` is mainly
  useful for comparing estimated against routed distances.
- The journey tables' `dist_km` is something else, a respondent's estimate of
  the whole journey: see Journeys.

### Modes

- `mode` (legs) uses one vocabulary for all years:
  - `motorbike_small` includes mopeds.
  - `transit_bus` includes PostBus and school buses.
  - `aerialway` covers cable cars, ski and chair lifts, funiculars and rack
    railways.
  - `taxi` includes ride-hailing (2021).
  - `micro` covers scooters, skates and similar.
  - The journeys' `mode_main` and `mode_back` use the same categories.
- **Bikes**: the surveys only tell e-bikes apart from 2015
  (`ebike25` = up to 25 km/h, `ebike45` = up to 45 km/h, number plate). Before
  that, e-bikes are counted as bikes:
  - 2010 → `anybike`, since e-bikes were already common enough to matter.
  - Up to 2005 → `rbike`, since they were still rare.
- `is_passenger`: for cars and motorbikes only (1 passenger, 0 driver).
  `car_occupancy` counts everyone in the car, including the respondent.
- **`mode_main_ranked` vs `mode_main_longest`** (trips): both are recomputed
  from the legs for every year, because the BFS main-mode variables differ
  between years (1994 by distance, 2000 none, 2005–2015 by ranking, 2021 both).
  - `mode_main_ranked` takes the highest-ranked mode used on the trip, in the
    BFS order: plane, rail, boat, tram, bus, aerialway, coach, car, truck,
    taxi, motorbike, small motorbike / moped, e-bike, bike, walk, micro,
    other. One train leg makes it a rail trip, whatever its length.
  - `mode_main_longest` is the mode of the leg with the largest `dist_km`.
  - Both are empty if any leg's mode is unknown (e.g. a pseudo leg).
    `mode_main_longest` is also empty if any leg's distance is unknown.

  **Which to use**: `mode_main_ranked` to match the BFS's long-standing main-mode
  definition; `mode_main_longest` when the question is which mode carried most
  of the distance.

### Purposes

- `purpose` (legs) is the activity at the leg's destination; "change of mode"
  (`mode_transfer`) marks legs within a trip.
- `purpose` (trips) is the BFS trip purpose. From 2000 the BFS gives a trip home
  the purpose of the longest activity of that outing, so `return` is rare in
  trips. Whether a trip goes out or back is in `direction`:
  - `outbound`: the destination is not home.
  - `return`: back home or to the accommodation.
  - `round`: from home to home, e.g. a walk.
- Shopping and errands are only told apart from 2005 (`errands_groceries` is all
  shopping, despite its name; `errands_services` is errands and services).
  1994 and 2000 have `errands_unspecified` instead. Merge the three when
  comparing across years.
- `accompany` exists from 2000; in 1994 such trips are under other purposes.
- 1994 has no `return` purpose at all: trips back carry the purpose of the
  outing, and only `direction` says they go home.
- `border_crossing`: the leg ends at the Swiss border (2005 on).
- `leisure_activity`: from 2010 in 17 kinds (e.g. `hiking`, `cycling`,
  `visit`). 2005 has no separate hiking and cycling. 2000 has 9 kinds, on trips
  only.
- Journey purposes have their own categories: see Journeys.

### People

- **Licences**: under-18s were not asked and are set to 0.
- **Transit passes**:
  - `pass_ga` (GA travelcard) and `pass_halbtax` (half-fare card) exist in all
    years.
  - `pass_season` is any other season ticket (regional zone pass, route pass,
    Gleis 7, junior card, other) and is comparable across all years.
  - `pass_regional` singles out the regional zone pass (*Verbund-Abo*), only
    from 2010.
  - 1994 and 2000 have no list of pass types: `pass_season` there means
    holding a weekly, monthly or annual pass (route and zone passes). In 1994,
    GA was only asked of people with some pass besides half-fare; it is 0 for
    the rest.
  - 2005 asked one combined question; its "other combinations" answer leaves
    GA and half-fare empty.
  - From 2010, passes only asked at certain ages (e.g. Gleis 7 at 16–25)
    count as not held in `pass_season` for everyone else.
- **`education`**: highest completed level, in six steps.
  - `tertiary_academic` includes colleges of higher education (HF, HTL, HWV),
    because 2000 does not tell them apart from universities of applied
    sciences.
  - 1994 asked yes/no per level; the highest "yes" is used.
- **`employment`**: up to 2005 it is the respondent's own main activity; from
  2010 it is the BFS labour-force status. The categories line up, but the
  concept differs (e.g. a student with a side job is `in_education` before 2010
  and `employed` after). Apprentices are `in_education` in all years.
  `employed_fulltime` comes from a percentage in 2021 (≥ 90 % = full-time).
- **Under 15**: `education`, `employment` and `employed_fulltime` are empty for
  target persons under 15 in every year. From 2010 the survey does not ask
  them; earlier years have answers for children (mostly "in education", or no
  completed education), which would otherwise make shares jump between years.
- **`weather`**: the first weather mentioned for the survey day. 2005 asked a
  different question with fewer categories (no `partly_cloudy` or `foggy`).
- **`survey_date`**: in 1994 it is derived from an undocumented day number
  (day 1 = 1 Jan 1994). It is used because its weekdays match the reported
  weekday for 98.8 % of people. The 1994 survey ran to 30 Jan 1995.
- `is_mobile`: 1 if the person left home on the survey day (1994: also if they
  spent the whole day abroad).

### Households

- **Income**: `income_min_chf` and `income_max_chf` bound the gross monthly
  household income band, in nominal CHF (not adjusted for inflation). The top
  band is open (`income_max_chf` empty): above 14,000 in 2000, above 16,000
  from 2005.
- `hh_type`: 1994 and 2000 only know "family", without telling couples with
  children from single parents.
- `n_bikes_any` includes e-bikes in all years; from 2015 the e-bikes are also
  counted separately.

### Household members

- Everyone living in the household, with age, sex and licences (and passes in
  2000 and 2005). `member_id` is `<hh_id>_m<member no.>`.
- `person_id` marks the members who are target persons. Up to 2010 the data
  says which members were selected; a selected member who was not interviewed
  (a second target person in some 2010 households) gets no `person_id`, so
  every `person_id` points to a row in `people`. From 2015 the data doesn't
  say: the target person is the respondent, and the questionnaire then asks
  for "the other people in the household". The member whose age and sex match
  the target person is used; the log reports how this was resolved (member 1
  in nearly all households, or a match by age and sex).

### Vehicles

- One row per car or motorbike. `vehicle_id` is `<hh_id>_<type><no.>`, e.g.
  `2015_123_car2`; the number counts within each type, as the leg question asks
  for "the 2nd car". Up to 2005 cars and motorbikes are numbered by km driven
  (1 = most used); from 2010 the order is not documented.
- `km_per_year` is the calendar year before the survey in 1994 and 2000, the
  last 12 months from 2005.
- `registration_month` was only asked for recently registered vehicles: 2000
  for 1999 on, 2005 for 2004 on (about 10 % filled). From 2010 it also comes
  from the registry, so most vehicles have it. 1994 has the year only.
- Engine size: `engine_cc_min` / `engine_cc_max`, like income. 1994 and 2000
  cars only give a band; later years an exact value (min = max). 2021 asked it
  for motorbikes only.
- **`fuel`**: none before 2005; 2005 only petrol / diesel / other; 2010 merges
  hybrid, E85 and gas (`alternative`). **Plug-in hybrids**: the survey has no
  such answer, so they are mostly reported as "hybrid", some as petrol or
  electric. They are identified from the registry data, whatever fuel was
  reported: in 2021, a car using both electricity and fuel (electricity
  consumption above zero, and fuel consumption or CO₂ above zero); in 2015, a
  car with CO₂ above 0 and below 70 g/km, which no ordinary hybrid or
  combustion car of that fleet reaches. The log shows the counts by reported
  fuel. Range extenders count as plug-in hybrids. Where a car was not found in
  the registry, a plug-in keeps its reported fuel (mostly `hybrid`).
- **Registry data** (MOFIS, from 2010): technical data from the national vehicle
  registry, matched via the licence plate. `in_registry` says whether the match
  worked; the registry columns are empty otherwise. `energy_label`,
  `emission_code` and `type_approval_no` are kept as raw codes; the emission
  code needs an external table to translate into a Euro emission standard.
- **`co2_g_km`** (2015 on), `fuel_l_100km` and `electricity_kwh_100km` (2021) are
  official type-approval values on the old NEDC test cycle, not measured in
  use. Real-world emissions are usually higher, and for plug-in hybrids often
  several times higher. The registry leaves them empty for battery-electric
  cars; they are set to 0 (tailpipe) for electric cars found in the registry,
  so fleet averages over `co2_g_km` include them.
- `main_driver_ids`: member ids of the main driver(s), `;`-separated; up to 1
  (1994, 2005), 2 (2010) or 5 (2015 on); for cars only (1994 also for
  motorbikes). 2000 did not ask. A main driver from outside the household (an
  answer in 2005 and 2010) is left out.

### Which car a leg used

- `car_source` (car legs, 2000 on): household car, company car, rental,
  car-sharing or other. For household cars, `vehicle_id` points to the row in
  `vehicles`. The car number was only asked in households with more than one
  car; with a single car it is that one. The log reports how many legs found
  their vehicle.
- **2000**: the car was not recorded in the first three months of fieldwork.
  Interviews from January to March 2000 have no `car_source` or `vehicle_id`
  (22 % of car legs); from April on, nearly every car leg has them (a few
  early-April interviews are still blank). Shares computed over the legs that
  have an answer therefore miss most of the winter.

### Journeys

- **What they are.** Two extra question modules (`extra_module`), each asked
  of about 30 % of the target persons: day journeys (*Tagesreisen*) in the
  14 days before the interview, and journeys with at least one night away in
  the last 4 months (2021: that ended in the last 4 months). Both leave out
  routine travel: a day journey lasts more than 3 hours (2021: at least 3),
  and neither may be made weekly or more often, or serve everyday needs. A
  long commute is not a day journey.
- **How they relate to trips and legs.** Trips and legs are one reference
  day. A day journey made on that day is already there, as ordinary trips.
  So is the return of an overnight journey that ends that day, but usually
  not its outbound part, because people are rarely reachable the day after
  they leave. The journey tables are a separate, longer look back at rare
  travel, recorded as whole journeys without legs or times of day. Don't add
  them to the trips: the BFS's annual mobility is the reference-day travel
  plus half the distance of the overnight journeys (their outbound parts). A
  day journey on the reference day appears in both and can be matched by its
  `date`.
- **One row per journey reported.** The detailed questions were asked for up
  to three journeys per person, picked at random (`is_detailed`). The other
  rows have only the date, from 2010 also `regularity` (and `nights` for
  overnight journeys). `n_day_journeys` and `n_overnight_journeys` in
  `people` are the counts the person gave.
- **Weights.** Use `weight` for counts over all journeys and
  `weight_detailed` (`weight` × journeys listed / journeys detailed) for
  anything from the detail columns. Per-person rates need the people who had
  the module as the base: Σ `weight` over the journeys / Σ `weight` over the
  people with that `extra_module`.
- `dist_km` (unlike in trips and legs) is the respondent's estimate for the
  whole journey: there and back, plus travel at the destination. From 2015
  the BFS imputed it where the respondent did not know (the version its main
  report uses). `dist_domestic_km` was only asked for journeys abroad; for
  journeys to a Swiss destination it is `dist_km`.
- `mode_main` is the mode covering the longest distance: in 2005 of the whole
  journey, from 2010 of the way out, with `mode_back` for the way back. 2005
  only asked whether the same mode was used back.
- **Purposes**: 2005 has one `sport`, which 2010 on splits into
  `sport_active`, `hiking`, `cycling` and `sport_passive`. 2005 has no
  `education`, `shopping` or `accompany_private`, and its own `unpaid_work`
  and `multiple`. Compare 2005 with later years on coarser groups.
- Start and destination are places the respondent named, geocoded to about
  municipality precision (so no geocode level). Places abroad: 2005 gives them
  in WGS84 degrees (Swiss places in LV03), converted like the later years; 2010
  has coordinates abroad only for the main destinations of overnight journeys,
  not for day-journey destinations or the places in `journey_places` (country
  only).
- **Other places** are in `journey_places`, up to three per detailed journey,
  linked by `journey_id`: for day journeys, other places with an activity of
  over 30 minutes (`place_type` `stop`); for overnight journeys, further
  important overnight places besides the main destination (`overnight`).
  Only their location is known, not when or how long. `has_places` says
  whether the person named any (2005 overnight journeys: not asked, the
  places were just listed).
- Not kept: flight details, reasons for the mode choice. 1994 had no journey
  questions.
- **2000 is left out, but could be added with a caveat.** A quarter of the
  target persons were asked about their *last* journey with an overnight stay,
  however long ago: when they came back, the nights, the main mode of the way
  back and the distance in Switzerland and abroad, plus how many other such
  journeys they made in the last 3 months. A separate table (RE) lists that
  journey's overnight places, with the mode to each and the purpose there.
  2000 has no day journeys. One "last journey" per person is a different
  sample from all journeys within a fixed period: each traveller counts once
  however often they travel, so shares describe travellers rather than
  journeys, and the interview date decides which season's journey is reported.
  Frequency is only known from the count of other journeys in 3 months (later
  years: 4). It would fit `overnight_journeys` as one detailed row per person,
  flagged as the last journey: usable for rough shares (modes, destinations),
  not for journey counts or rates comparable with later years.

### Locations

- `*_x`, `*_y` are LV95 metres (EPSG:2056) in all years. In households, trips
  and legs, `*_geocode_level` says how precise they are (`address`, `street`,
  `postcode`, `municipality`, `region`, `country`); journey places have no
  geocode level (see Journeys).
- **1994 and 2000 have no coordinates.** Their points are the BFS postcode
  reference points as of 1 January 2000, which come with the 2000 delivery
  (`MZ2000/Codes/Schweiz/CH_Ortschaften.xls`). Where the postcode is unknown,
  the municipality's point is used: the mean of its postcode points. Within a
  postcode area, all places share one point, so distances computed from these
  points mean little for short trips. The BFS notes that using these
  coordinates needs a GEOSTAT licence beyond the MTMC one.
- `*_municipality` is the BFS municipality number as of the survey year.
  Municipalities have merged since, so join them to boundaries of the same
  year, or use `municipality` in the context tables (today's municipalities,
  see [Areas](#areas)).
- `*_in_switzerland` comes from a different source each year:
  - 2010 on: the country code.
  - 2005: the country name.
  - 2000: the "abroad" postcode.
  - 1994 has no country information: 1 where the postcode or municipality is
    Swiss, empty otherwise.
- **Pseudo legs** (`is_pseudo`, trips: `has_pseudo_leg`) are legs the BFS
  recorded without details; their mode, times and distance are empty.

## Areas

`../b_mzmv_context.py` places every home (`household_context.csv`) and trip
origin (`trip_orig_context.csv`) in fixed areas, for averages by area and
their trends over time. Both tables have `municipality` (BFS number) and
`municipality_name`, `postcode`, `canton_no` and `canton` (abbreviation,
e.g. `ZH`), `urban_character`, `agglomeration_size`, `language_region` and
`pool` (language region and urban character combined, e.g. `fr_core_city`).
They also hold the densities around the point (see [Densities](#densities)).

- **Fixed areas.** Every year's points are joined to the same, current
  polygons: municipalities as of 1 January 2023 and today's postcodes. A place
  gets the same areas in every year, even where municipalities merged or
  postcodes changed since. Likewise, `urban_character` and `language_region`
  are one fixed BFS classification (2020) for all years, not the typologies
  the survey files carry (the 2000 version up to 2010, the 2012 one from
  2015). Trends therefore describe places as they are classed in 2020.
- **`urban_character`** (BFS *Raum mit städtischem Charakter 2020*):

  | Class | BFS name |
  |---|---|
  | `core_city` | Agglomerationskerngemeinde (Kernstadt) |
  | `main_core` | Agglomerationskerngemeinde (Hauptkern) |
  | `secondary_core` | Agglomerationskerngemeinde (Nebenkern) |
  | `belt` | Agglomerationsgürtelgemeinde |
  | `multi_oriented` | Mehrfach orientierte Gemeinde |
  | `rural_centre` | Kerngemeinde ausserhalb Agglomerationen |
  | `rural` | Ländliche Gemeinde ohne städtischen Charakter |

- **`agglomeration_size`** (BFS 2020 size class, inhabitants of the whole
  agglomeration): `500k_plus` (Zurich, Geneva, Basel), `250k_500k` (Bern,
  Lausanne), `100k_250k`, `50k_100k`, `under_50k`, or `outside`. Half the
  people in `core_city` live in the six largest cities, the other half in
  small-town cores (e.g. Bulle, Altdorf, Interlaken), so `core_city` and
  `main_core` are worth splitting by it.
- **`language_region`**: `de`, `fr`, `it`, `rm`. The Romansh region is only
  15 municipalities; merge it with `de` for most uses.
- 1994 and 2000 homes and origins are postcode points, so their postcode and
  municipality are only as good as the postcode. Points just outside a
  polygon (borders, lake shores) take the nearest one within 1 km. Trip
  origins abroad get no areas.
- **Sample sizes.** Target persons per area, from a count over all years:
  - municipalities with at least 100 in every year: 10 (1994–2021), 33
    (2005–2021), 71 (2010–2021);
  - postcodes: 2, 31 and 73;
  - NPVM traffic zones: none (median 4–7 people per zone from 2005);
  - so: municipalities for 2010–2021 and the larger cities; pools (language
    × urban character) for the full series. The log lists the target
    persons per pool and year.
  - Pools with at least 200 target persons in all six years (1994 is the
    smallest sample): all seven German-speaking ones; `fr_core_city`,
    `fr_main_core`, `fr_belt`, `fr_rural`; `it_core_city` (200 in 1994).
    The other French and Italian pools and `rm_rural` need merging, e.g.
    Romansh with German, Italian into cores (core city, main and secondary
    core) and the rest, `fr_rural_centre` with `fr_rural`.
- The joined municipality equals the survey's own number (where it still
  exists) for 100 % of homes in 2015 and 2021, 98–100 % in 2005 and 2010,
  and 97–98 % in 1994 and 2000 (postcode points).
- Sources: `raw/switzerland/bfs_spatial_levels/` (BFS municipality
  classifications) and `raw/switzerland/postcodes/` (swisstopo postcode
  directory), each with a `source.txt`.

## Densities

Residents and jobs per hectare, around every home and trip origin and for
whole municipalities and postcodes. They come from the LUMOS hectare data:
residents and full-time equivalent jobs (`fte`, all three sectors) on the
446,995 hectares of the Swiss settlement area, 1990–2025 in 5-year steps. These tables hold only sums over
circles and areas, no single hectares. Multiply by 100 for densities per km².

### Denominators

Every density comes with three denominators:

| Denominator | Hectares counted | Reads as |
|---|---|---|
| `occupied` | with residents or FTE in that year | how intensely the used land is used: infill raises it, building on new land can lower it |
| `mask` | of the LUMOS settlement area, the same set in every year | a fixed denominator: changes over time are changes in the counts alone |
| `gross` | all land: inside a municipality polygon, so no large lakes and nothing abroad | the usual density per land area; it includes forest, farmland and mountains, so it largely measures how much of the area is built up |

Residents and FTE share the denominators, so the two densities add up to
the density of residents and FTE together. A hectare counts by its centre.

### Around homes and trip origins

`household_context.csv` and `trip_orig_context.csv` have 12 density
columns, `<measure>_per_ha_<denominator>_<radius>`:

- measure `residents` or `fte`, the three denominators, radius `1km` or
  `3km`; e.g. `residents_per_ha_occupied_1km`, `fte_per_ha_gross_3km`;
- the circle holds the hectares whose centre is within the radius of the
  centre of the point's hectare: 317 hectares for 1 km, 2,821 for 3 km;
- the hectare data is from the LUMOS year closest to the survey: the same
  year, but 1995 for 1994 and 2020 for 2021;
- empty for points located only to a postcode or coarser (`*_geocode_level`
  `postcode`, `municipality`, `region` or `country`; that is all of 1994 and
  2000), since a circle around such a point is not the home's surroundings.
  Use the area densities there. From 2005 this leaves out 3 % of homes in
  2005 (under 1 % later) and 7 % of the trip origins in Switzerland in 2005
  (1.5–2.5 % later), mostly in sparser places. Also empty for trip origins
  abroad and points without coordinates;
- near the border, circles cover only the Swiss side, in the counts and in
  the `gross` land alike.

### Of municipalities and postcodes

`municipality_density.csv` and `postcode_density.csv` have one row per area
and LUMOS year (1990, 1995, …, 2025):

- `municipality` (with `municipality_name`) or `postcode` and `year`;
- the totals `residents` and `fte`, the hectares `ha_occupied`, `ha_mask`
  and `ha_gross`, and the six densities `<measure>_per_ha_<denominator>`.
- The areas are the fixed ones of the context tables, so an area's series
  keeps the same extent in every year. A hectare belongs to the area
  containing its centre, or else the nearest within 1 km (lake shores,
  borders).
- Join by `municipality` or `postcode` from the context tables and the
  survey's LUMOS year (see above). With 1994 → 1995 and 2021 → 2020, that
  gives every survey year a density, also 1994 and 2000.
- To pool areas, add up the totals and hectares, then divide. Don't average
  the densities.

### Comparability over time

- **FTE.** The LUMOS FTE change source between 2005 and 2010 (business
  census, then STATENT; the 2010 hectares hold 2011 data). Against BFS's
  consistent STATENT series, growth from 2005 to 2010 is inflated by about
  10 % (primary sector 26 %, secondary 2 %, tertiary 13 %), and unevenly:
  suburban and rural municipalities gain several points against core cities,
  which neither a national nor a per-sector factor removes. Small values are
  also treated differently in the two eras (to 2005, hectares with 1–3 FTE
  hold 3; from 2010 no hectare holds 1–3), which inflates sparse places,
  above all the primary sector (1.6× the BFS total in 2005, 2.0× in 2010).
  FTE densities are therefore not comparable across that break, and nor are
  changes of residents and FTE together. BFS also changed its FTE method in
  2015. 1990 reuses the 1995 FTE.
- **Residents in 1995 and 2005 (cause found 2026-11-06).** LUMOS totals
  6.78 M residents in 1995 and 7.28 M in 2005, about 4 % and 2.4 % below the
  official population (about 7.06 M and 7.46 M), and below its own 1990 total
  (6.89 M). The other years match the official figures. The cause is not the
  BFS files: LUMOS's 1995 and 2005 population rasters are halfway
  interpolations, (1990 + 2000) / 2 and (2000 + 2010) / 2, computed only where
  both years have data. That drops 52k / 62k hectares and 208k / 176k
  residents; aperta-lumos's integer CSVs also round the halves down. The
  shortfall is therefore uneven: it falls on hectares populated in only one of
  the two bracketing years, so it adds error to changes from or to 1995 and
  2005, not just to their level.
- **Settlement mask.** LUMOS keeps only hectares of its settlement mask,
  which covers everything from 1995 to 2020 but leaves out 50k residents in
  1990 (0.7 %) and, in 2025, 10k residents and 39k FTE (0.8 %), presumably
  hectares built on after 2020. This affects the 1990 and 2025 rows of the
  area density tables, not the survey years.

## Data quirks handled in the codebooks

- 1994: three legs appear twice, identical in every field (household 9404).
  The duplicates are dropped, with a warning in the log.
- 2000: leg purpose code 9 is "no answer" in the database description but
  "return home" in the questionnaire. It is on 28 % of legs, about the share
  of return legs the BFS describes, so it is read as return home.
- 2000: leisure activity code 0 (a few dozen trips) is undocumented and left
  empty.
- 2005: the coordinates' system is not stated. They are LV03, and the X/Y axis
  order is detected from the values. The journey tables give places abroad in
  WGS84 degrees instead; these are told apart by their values.
- 2005: the description of the trip direction swaps codes 2 and 3 in its text.
  The value labels are used; they agree with the 2015 description.
- 2005: registration month code 13 ("don't know") is left empty.
- 2005, 2010: main-driver code 16 ("another person") is left out.
- 2010: the target-person number in the household-member table marks members
  *selected* as target persons; 6,352 of them were never interviewed and get
  no `person_id`.
- 2015: the registry CO₂ is labelled g/100 km, but the values are g/km (the
  log checks the median).
- 2015, 2021: the registry has no CO₂ value for battery-electric cars; it is
  set to 0 for those found in the registry.
- 2015, 2021: journey-place coordinates use −997 and −999 as missing codes.
