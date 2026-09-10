# Regressie-rapport — versgewichtschatting

Ridge-regressie op vorm-features uit segmentatiemaskers → versgewicht (g).
**Getraind op synthetische data** (module 1); hertrain op echte gewogen data.

- GSD gebruikt: **0.25 cm/px**
- Train/test split: **360** / **120**

## Metrics (testset)

| Metric | Waarde |
|--------|--------|
| R² | 0.940 |
| MAE (g) | 30.9 |
| Klasse-accuraatheid (<350 / 350-500 / >500 g) | 91.7% |

## Voorspelde klasse-verdeling (testset)

| Klasse | Aantal |
|--------|--------|
| light | 55 |
| medium | 31 |
| heavy | 34 |

Klassegrenzen: `light < 350 g`, `medium 350-500 g`, `heavy > 500 g`.
