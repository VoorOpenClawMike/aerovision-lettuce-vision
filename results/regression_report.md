# Regressie-rapport — versgewichtschatting

Ridge-regressie op vorm-features uit segmentatiemaskers → versgewicht (g).
**Getraind op synthetische data** (module 1); hertrain op echte gewogen data.

- GSD gebruikt: **0.25 cm/px**
- Train/test split: **112** / **38**

## Metrics (testset)

| Metric | Waarde |
|--------|--------|
| R² | 0.922 |
| MAE (g) | 29.0 |
| Klasse-accuraatheid (<350 / 350-500 / >500 g) | 86.8% |

## Voorspelde klasse-verdeling (testset)

| Klasse | Aantal |
|--------|--------|
| light | 16 |
| medium | 14 |
| heavy | 8 |

Klassegrenzen: `light < 350 g`, `medium 350-500 g`, `heavy > 500 g`.
