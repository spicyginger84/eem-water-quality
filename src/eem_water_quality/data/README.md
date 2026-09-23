# Data layer

`io.py` loads the processed EEM contract, `schema.py` defines column names and
validation, and `splitting.py` contains compatibility one-shot split helpers.
The benchmark CV splitters live in `eem_water_quality.evaluation` because they
are evaluation protocols rather than data loading concerns.

