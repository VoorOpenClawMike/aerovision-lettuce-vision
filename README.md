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

---

## Module 3 — GSD-calibratie (`src/calibration/gsd.py`)

Berekent de Ground Sampling Distance en zet pixel-oppervlak om naar cm².

```
GSD [cm/px] = (sensorbreedte_mm · vlieghoogte_m · 100) / (brandpuntsafstand_mm · beeldbreedte_px)
area_cm2    = area_px · GSD²
```

> **Correctie t.o.v. opdrachtformule:** de opdracht schreef de formule zonder de
> factor `·100`. Zonder die factor is de uitkomst *meter* per pixel i.p.v.
> *cm* per pixel (100× te klein). Deze module gebruikt de fysisch correcte vorm:
> een DJI Phantom 4 Pro op 30 m geeft dan ~0.82 cm/px, wat overeenkomt met de
> fabriekstabellen. De testsuite pint deze waarde.

Ingebouwde camera-presets (`dji_phantom4_pro`, `dji_mavic3`, `generic_1_2_3`).
Meerdere vlieghoogtes geven simpelweg verschillende GSD's (hoger → grover).

**Gebruik:**
```bash
python -m src.calibration.gsd --preset dji_phantom4_pro --altitude-m 30 --area-px 5000
```

**Echte data:** lees `sensorbreedte`, `brandpuntsafstand` en `beeldbreedte` uit
de EXIF/specsheet van de gebruikte drone en de `vlieghoogte` uit het vluchtplan.

**Tests:** `tests/test_gsd.py` (referentiewaarde, lineaire hoogte-schaling,
oppervlak-schaling met GSD², invoervalidatie).

---

## Module 2 — Detectie / segmentatie (`src/detection/`)

YOLO11-seg (Ultralytics) trainings- en inferentie-wrapper, plus een
COCO→YOLO-seg-conversie.

- **`coco_to_yolo.py`** — converteert `instances.coco.json` naar de
  Ultralytics-mappenstructuur (`images/{train,val}`, `labels/{train,val}`,
  `data.yaml`), met genormaliseerde polygon-labels en een deterministische
  80/20-split. Puur (stdlib), volledig unit-getest.
- **`train.py`** — trainingswrapper, configureerbare `epochs`/`imgsz`/`batch`/
  `device`. Default **50 epochs op CPU** op de synthetische set (draait in
  enkele minuten). `build_train_config()` is puur en getest.
- **`infer.py`** — draait het model over beelden, zet elk instance-masker om in
  een detectierecord (centroïde + `area_px`, optioneel `area_cm2`) en schrijft
  `results/detections.csv`. Masker-helpers zijn pure NumPy en getest.

**Gebruik:**
```bash
python -m src.detection.coco_to_yolo --coco data/synthetic/annotations/instances.coco.json \
    --images data/synthetic/images --out data/yolo_seg
python -m src.detection.train --data data/yolo_seg/data.yaml --epochs 50 --device cpu
python -m src.detection.infer --model runs/segment/krop_seg/weights/best.pt \
    --images data/synthetic/images --out results/detections.csv --gsd 0.25
```

**Opschalen naar productie (echte GPU / Colab):** gebruik een groter model
(`yolo11m/l/x-seg.pt`), `--device 0`, `epochs 100–300`, grotere `batch` en
`imgsz 1024` voor hoge-resolutie orthomosaïeken. Zie `build_train_config`.

**Tests:** `tests/test_detection.py` (normalisatie/clamping, split,
COCO→YOLO-layout, masker-area/centroïde, resize, trainconfig-validatie).

---

## Module 4 — Gewichtsregressie (`src/regression/weight_model.py`)

Uit elk segmentatiemasker worden vorm-features berekend
(`area_cm2`, `perimeter_cm`, `compactness = 4π·A/P²`, `hull_ratio = A/hull`,
`eq_diameter_cm`) en via **Ridge-regressie** (scikit-learn, met
`StandardScaler`) omgezet naar geschat versgewicht. Daarna classificatie in
`light`/`medium`/`heavy`.

Features komen uit polygonen (COCO) óf uit maskers (cv2-contour), zodat exact
dezelfde feature-pijplijn geldt voor de synthetische annotaties én voor
YOLO-voorspellingen.

**Resultaat op de synthetische set** (480 kroppen, `results/regression_report.md`):
R² ≈ **0.94**, MAE ≈ **31 g**, klasse-accuraatheid ≈ **92 %**. Dit toont dat de
pijplijn de (synthetische) vorm→gewicht-relatie leert; met echte gewogen data
moeten deze cijfers opnieuw worden vastgesteld.

**Gebruik:**
```bash
python -m src.regression.weight_model --gsd 0.25 --report results/regression_report.md
```

**Tests:** `tests/test_weight_model.py` (feature-correctheid op cirkel/vierkant,
flat vs pairs, mask-features, fit/predict/klassen, end-to-end op synth-set).

---

## Module 5 — Steekproef-statistiek (`src/sampling/stats.py`)

Bepaalt welk steekproefpercentage (10/25/50 %) een betrouwbare schatting van de
gewichtsklasseverdeling geeft.

- **Cochran's formule** `n0 = z²·p(1-p)/e²` + **finite population correction**
  `n = n0/(1+(n0-1)/N)`. Default `p=0.5` (meest conservatief).
- **Bootstrap-BI (BCa)** via `scipy.stats.bootstrap` op de klasseproporties per
  kandidaat-fractie — de empirische tegenhanger die laat zien hoe de BI-breedte
  krimpt bij grotere steekproeven.

Voorbeeld: voor `N=5000`, 95 % betrouwbaarheid, 5 % marge is de vereiste
steekproef **n=357** (≈7 %), dus **10 %** volstaat.

**Gebruik:**
```bash
python -m src.sampling.stats --population 5000 --confidence 0.95 --margin 0.05 \
    --labels-csv data/synthetic/ground_truth_weights.csv
```

**Tests:** `tests/test_sampling.py` (z-waarden, worst-case n≈384, FPC, aanbeveling,
BCa-BI-grenzen en degeneratie, fractie-evaluatie).

---

## Pijplijn-orkestrator (`src/pipeline.py`)

Verbindt detectie → features → regressie → klasse en schrijft
`results/field_detections.csv` (`krop_id, cx, cy, area_cm2, weight_g,
weight_class, lat, lon`) — de invoer voor het dashboard.

- Gebruikt **YOLO-detecties** als een getraind model beelden detecteert; anders
  valt het terug op de **COCO-grondwaarheidspolygonen** (duidelijk gelogd), zodat
  de demo altijd data heeft.
- **Synthetische geo-referentie:** beelden hebben geen echte wereldcoördinaten;
  ze worden op een raster nabij Midden-Limburg (≈51.25 N, 5.95 E) geplaatst.
  Echte orthomosaïeken (GeoTIFF) dragen wél wereldcoördinaten — vervang
  `pixel_to_latlon` dan.

```bash
python -m src.pipeline --gsd 0.25   # + optioneel --model runs/.../best.pt
```

---

## Module 6 — Streamlit-dashboard (`src/dashboard/app.py`)

Laadt `results/field_detections.csv` en toont:
(a) **aantallen per klasse** (KPI's + gemiddeld gewicht + geschatte opbrengst),
(b) een **kleurgecodeerde folium-veldkaart** (amber `light` / groen `medium` /
roestrood `heavy` — Bed & Bracket-kleuren), en
(c) een **gewichtshistogram** per klasse.

Pure data-helpers staan in `src/dashboard/data.py` (getest); `app.py` bevat de
dunne Streamlit-UI.

**Gebruik:**
```bash
python -m src.pipeline                       # maakt results/field_detections.csv
streamlit run src/dashboard/app.py           # open de demo
```

**Tests:** `tests/test_dashboard.py` (CSV-validatie, klassetellingen,
kleurmapping, KPI's, histogram-bins, folium-kaartopbouw).

---

## Module 7 — Edge-export (`src/edge/export_tensorrt.py`)

Exporteert het getrainde YOLO-model naar **ONNX** (draait hier) en documenteert
de **TensorRT-INT8**-conversie die alleen op een **Jetson met JetPack** kan
(niet in deze container). De volledige Jetson-stappen staan in
`TENSORRT_INSTRUCTIONS` en worden weggeschreven naar `src/edge/TENSORRT.md`.

> **Belangrijk:** INT8-kalibratie vereist een *representatieve set echte*
> orthomosaïek-crops — synthetische data geeft geen productie-representatieve
> kwantisatie.

**Gebruik:**
```bash
python -m src.edge.export_tensorrt --model runs/.../best.pt --imgsz 640 --write-trt-readme
```

**Tests:** `tests/test_edge.py` (ontbrekend-model-fout, instructie-inhoud,
README-writer, echte ONNX-export indien een model beschikbaar is).
