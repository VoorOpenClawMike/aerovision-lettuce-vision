# RESULTS — Aerovision / SOIL-Data krop-sla pipeline

Samenvatting van wat werkt, wat gesynthetiseerd is (en dus met echte data
opnieuw moet), en de vervolgstappen. Zie `README.md` per module en
`BLOCKERS.md` voor omgevingsbeperkingen.

## Wat werkt (geverifieerd end-to-end)

| # | Module | Status | Bewijs |
|---|--------|--------|--------|
| 1 | `data/synth_generator` | ✅ | 16 beelden / **480 kroppen**, COCO-json + gewicht-CSV; 5 tests |
| 2 | `detection` (YOLO11-seg) | ✅ | COCO→YOLO-conversie (13 train/3 val), CPU-training draait, ONNX-export werkt; 6 tests |
| 3 | `calibration/gsd` | ✅ | fysisch-correcte GSD (P4P@30m ≈ 0.82 cm/px); 5 tests |
| 4 | `regression/weight_model` | ✅ | Ridge: **R² ≈ 0.94, MAE ≈ 31 g, klasse-acc ≈ 92 %**; 6 tests |
| 5 | `sampling/stats` | ✅ | Cochran+FPC+BCa; N=5000/95%/5% → n=357 (10 %); 6 tests |
| 6 | `dashboard` (Streamlit) | ✅ | KPI's + folium-veldkaart + histogram; 6 tests |
| 7 | `edge/export_tensorrt` | ✅ | ONNX-export + gedocumenteerde Jetson-INT8-stappen; 4 tests |
| – | `pipeline` (orkestrator) | ✅ | detectie→features→gewicht→klasse→CSV (geo); 3 tests |

**Testsuite:** `python -m pytest -q` → **41 passed**.

**Detectiepad geverifieerd:** de orkestrator draaide zowel via het echte
YOLO-model (`detection_source: "yolo"`, 175 detecties op een tussentijds
checkpoint) als via de grondwaarheid-fallback (480 detecties, alle klassen).

## Wat synthetisch is (moet met echte data opnieuw)

Alles wat op grondwaarheid steunt is **synthetisch gegenereerd** en moet met
echte, gewogen drone-data opnieuw worden getraind/gevalideerd:

1. **Beelden + maskers** — nu procedureel getekende blobs. → Echte
   orthomosaïeken + handmatige/ge-assisteerde annotatie (CVAT/Roboflow) in
   hetzelfde COCO-formaat.
2. **Gewicht-grondwaarheid** — nu `weight = 2.2·area_cm2^0.85 · ruis`. → Echte
   veldweeg-metingen per krop. De regressie-metrics (R²/MAE) hierboven meten
   dus vooral dat de pijplijn een *bekende* synthetische relatie terugvindt; ze
   zeggen nog **niets** over echte nauwkeurigheid.
3. **YOLO-detectiegewichten** — kort op CPU getraind op weinig data. → Fatsoenlijk
   trainen op GPU/Colab met een groter model (`yolo11m/l/x-seg`) op echte data.
4. **Geo-referentie** — nu een synthetisch raster bij Midden-Limburg. → Echte
   wereldcoördinaten uit de GeoTIFF-orthomosaïek (`pixel_to_latlon` vervangen).
5. **GSD-parameters** — nu defaults/presets. → Echte sensor/brandpunt/hoogte uit
   EXIF + vluchtplan.

## Vervolgstappen (productie)

1. **Data verzamelen & annoteren:** vlieg meerdere hoogtes, maak
   orthomosaïeken, annoteer een representatieve set kroppen (COCO-seg), weeg een
   subset fysiek voor gewicht-grondwaarheid.
2. **Detector trainen:** `yolo11m-seg` op GPU, `epochs 100–300`, `imgsz 1024`,
   augmentaties; valideer mask-mAP op een aparte veldstrook.
3. **Regressie herijken:** hertrain `weight_model` op echte features↔gewicht;
   rapporteer R²/MAE per vlieghoogte; overweeg niet-lineaire modellen als Ridge
   tekortschiet.
4. **Steekproefbeleid vaststellen:** draai `sampling/stats` op de echte
   populatiegrootte om het inspectiepercentage (10/25/50 %) te fixeren.
5. **Edge-deployment:** exporteer naar ONNX (klaar) en bouw de TensorRT-INT8-
   engine op de Jetson met een echte kalibratieset (`src/edge/TENSORRT.md`);
   valideer mAP-verlies < 1–2 %.
6. **Dashboard uitrollen:** koppel aan de echte detectie-CSV; voeg
   perceel-/datumfilters en export toe.

## Reproduceren

```bash
pip install -r requirements.txt
bash run_pipeline.sh              # data → (train) → regressie → pijplijn → steekproefadvies
streamlit run src/dashboard/app.py
python -m pytest -q               # 41 tests
```
