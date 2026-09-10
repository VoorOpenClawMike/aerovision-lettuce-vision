# Aerovision / SOIL-Data — Drone-vision pipeline voor kroppen sla

End-to-end computer-vision pipeline die op **drone-RGB-orthomosaïeken** kroppen
sla detecteert (instance-segmentatie), hun **versgewicht** schat, en per veld een
**gewichtsklasseverdeling** rapporteert met een statistisch onderbouwd
steekproefadvies. Gebouwd als Inholland-minorproject voor SOIL-Data (via
Aerovision), Midden-Limburg.

> **Belangrijk:** er is op dit moment nog **geen echte geannoteerde, gewogen
> dataset**. De volledige pipeline draait daarom op een **synthetische
> datagenerator** (module 1) zodat alles end-to-end te testen is. Overal waar
> synthetische data gebruikt wordt, staat gedocumenteerd hoe echte data wordt
> aangesloten. Zie ook [`RESULTS.md`](RESULTS.md) en [`BLOCKERS.md`](BLOCKERS.md).

## Pipeline-overzicht

```
 (1) synth_generator ──▶ PNG + COCO-json + gewicht-CSV
        │
        ▼
 (2) detection (YOLO11-seg)  ◀── COCO→YOLO-seg conversie
        │  maskers per krop
        ▼
 (3) calibration/gsd  ──▶ pixel-oppervlak → cm² (vluchtparameters)
        │
        ▼
 (4) regression/weight_model ──▶ features → Ridge → gewicht + klasse (<350/350-500/>500 g)
        │
        ├──▶ (5) sampling/stats ──▶ steekproefadvies (Cochran + FPC + bootstrap-BI)
        │
        └──▶ (6) dashboard (Streamlit + folium veldkaart)

 (7) edge/export_tensorrt ──▶ ONNX-export (+ TensorRT-INT8 TODO op Jetson)
```

## Installatie

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# In een headless container: opencv-python-headless (staat in requirements.txt).
```

Snelle end-to-end demo (synthetische data → training → regressie → dashboard):

```bash
bash run_pipeline.sh          # genereert data, converteert, traint kort, evalueert
streamlit run src/dashboard/app.py
```

## Gewichtsklassen (consistent door de hele pipeline)

| Klasse   | Gewicht        | Dashboard-kleur (Bed & Bracket) |
|----------|----------------|---------------------------------|
| `light`  | `< 350 g`      | amber                           |
| `medium` | `350 – 500 g`  | groen                           |
| `heavy`  | `> 500 g`      | roestrood                       |

---

## Module 1 — Synthetische datagenerator (`src/data/synth_generator.py`)

Genereert stand-in "drone-orthomosaïek"-tegels met blob-vormige kroppen op een
getextureerde grond-achtergrond, plus bijbehorende grondwaarheid.

**Outputs** (in `data/synthetic/`):
- `images/orthomosaic_XXX.png` — RGB-tegels (default 640×640).
- `annotations/instances.coco.json` — COCO instance-segmentatie; één polygon
  per krop in het `segmentation`-veld.
- `ground_truth_weights.csv` — per krop: `area_px`, `area_cm2`, `weight_g`,
  `weight_class`.

**Synthetisch gewichtsmodel** (gedocumenteerde keuze, herzien met echte data):
`weight_g = 2.2 · area_cm2^0.85 · lognormaal-ruis(σ=0.10)`, met een nominale
GSD van `0.25 cm/px`. De sub-lineaire exponent modelleert dat geprojecteerd
bladoppervlak verzadigt naarmate koppen groeien.

**Gebruik:**
```bash
python -m src.data.synth_generator --num-images 6 --heads-per-image 25 --seed 42
```

**Echte data aansluiten:** vervang de `data/synthetic/`-outputs door echte
orthomosaïeken + een echte COCO-annotatie (bv. uit CVAT/Roboflow) en een echte
weeg-CSV. De rest van de pipeline leest exact hetzelfde formaat; niets anders
hoeft te veranderen.

**Tests:** `tests/test_synth_generator.py` (geometrie, gewichtsmodel,
end-to-end generatie, determinisme).
