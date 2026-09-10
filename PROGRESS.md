# Voortgangslog

Doorlopend logboek van de autonome build. Nieuwste onderaan.

- **Setup** — repo geïnspecteerd; deps geverifieerd (numpy/PIL/sklearn/scipy/
  pandas/streamlit/folium aanwezig). `opencv-python-headless`, `pytest`,
  `shapely` geïnstalleerd (headless container mist `libGL.so.1`, daarom de
  headless OpenCV-build). Git geïnitialiseerd op branch `main`.
- **Module 1 — synth_generator** ✅ Gebouwd + 5 unit tests groen. Genereert
  6 tegels / 150 kroppen met klasse-spreiding light/medium/heavy = 79/36/35.
  COCO-json + gewicht-CSV gevalideerd. README-sectie toegevoegd.
- **Module 3 — calibration/gsd** ✅ Gebouwd + 5 unit tests groen. Fysisch
  correcte GSD-formule (×100 m→cm; opdrachtformule miste die factor — expliciet
  gedocumenteerd). Camera-presets + CLI. README-sectie toegevoegd.
- **Module 2 — detection** ✅ COCO→YOLO-seg converter + train/infer wrappers
  gebouwd; 6 unit tests groen. Converter geeft 5 train / 1 val / 150 instances.
  Echte YOLO11n-seg training gestart op CPU (yolo11n-seg.pt download via mirror;
  github.com is geblokkeerd in de container). README-sectie toegevoegd.
- **Module 4 — regression/weight_model** ✅ Gebouwd + 6 unit tests groen.
  Ridge-regressie op 5 vorm-features. Op synth-set: R²≈0.92, MAE≈29 g,
  klasse-accuraatheid≈87%. Rapport naar results/regression_report.md.
  README-sectie toegevoegd.
