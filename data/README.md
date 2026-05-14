# Data

This folder is where the raw ELSA-Brasil MSK CSV is expected to live.

## `raw/stataToCsvMG.csv` (not committed)

The analytical input is `data/raw/stataToCsvMG.csv`, a 232-column export from
the ELSA-Brasil Musculoskeletal Study (2012-2014, baseline wave + selected
follow-up variables). **This file is access-controlled** and is not
distributed with the repository. To obtain it, follow the ELSA-Brasil data
request process at <https://www.elsa.org.br/>.

Once you have the file:

```
mkdir -p data/raw
cp /path/to/stataToCsvMG.csv data/raw/
```

`.gitignore` already excludes `data/raw/` to prevent accidental commits.

The first kilobyte of the CSV used to generate the published results has
SHA-256 prefix `15ef4e96a414cca6` (see `tests/fixtures/fixture_metadata.json`).
If your local copy hashes to the same value, the regression tests will pass
bit-for-bit.

## `codebook/`

- `variable_codebook.csv` — tidy export of the codebook sheet from the
  original ELSA workbook, mapping each STATA code (`b_kld`, `idadeb`, ...)
  to its analysis name, description, type, and ELSA wave. 165 rows, 9
  columns. Diffable in git, safe to publish.

> The original `Seleção de variáveis OAJ ELSA.xlsx` is **NOT** in this
> folder. That workbook contains two hidden sheets (`Knee X-Rays`, 3,115
> rows; `Sheet4`, 15,105 rows) with participant-level ELSA data including
> `idelsa` identifiers and is therefore access-controlled. Co-authors keep
> a copy under `data/raw/` (gitignored). `variable_codebook.csv` is what
> the public repo exposes; if you need the original workbook, request it
> through the ELSA-Brasil data process alongside the raw CSV.
