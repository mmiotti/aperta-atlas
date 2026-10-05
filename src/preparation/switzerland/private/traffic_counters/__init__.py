from aperta_atlas.context import Storage

# NPVM zaehldaten (Swiss ASTRA traffic counters) are not redistributable —
# `prepare_counters.py` reads them from `<DATA_DIR_PRIVATE>/raw/` and writes
# the prepared point gpkg back to PRIVATE.
STORAGE = Storage.PRIVATE
