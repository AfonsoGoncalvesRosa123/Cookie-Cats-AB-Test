# Data

Download `cookie_cats.csv` (about 2.8 MB, 90,189 rows) from Kaggle and put it in this folder:

https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats

The CSV is not committed to the repository (see `.gitignore`).

| Column | Meaning |
|---|---|
| `userid` | Player id |
| `version` | `gate_30` (control) or `gate_40` (treatment) |
| `sum_gamerounds` | Game rounds played in the first 14 days after install |
| `retention_1` | Player came back 1 day after install |
| `retention_7` | Player came back 7 days after install |
