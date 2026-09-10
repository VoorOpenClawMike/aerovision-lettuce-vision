# Bouwopdracht: Aerovision/SOIL-Data drone-vision pipeline voor kroppen sla

Je bent een autonome software-engineer. Bouw dit project VOLLEDIG en ZELFSTANDIG af.
Stel GEEN vragen, vraag NOOIT om bevestiging. Als een keuze open is, maak een
redelijke, goed gedocumenteerde technische keuze en ga door. Als iets echt
blokkeert (bv. een externe dataset die niet bestaat), schrijf dat naar
BLOCKERS.md en werk verder aan alle andere onderdelen — stop nooit volledig.

## Context
Klant: SOIL-Data (via Aerovision), Inholland-minorproject. Drone-RGB-beelden
van kroppen sla, meerdere vlieghoogtes, Midden-Limburg. Geen echte
geannoteerde dataset of gewogen grondwaarheid beschikbaar op dit moment —
bouw de volledige pipeline met een SYNTHETISCHE datagenerator zodat alles
end-to-end te testen is, en maak duidelijk in de README hoe echte data later
wordt aangesloten.

## Te bouwen modules (elk met unit tests + README-sectie)

1. `src/data/synth_generator.py` — genereert synthetische "drone-orthomosaïek"
   PNG's met random cirkelvormige/blob-vormige "kroppen" (verschillende
   groottes/kleuren/overlap) + bijbehorende COCO-json annotaties (polygon
   masks) en een CSV met synthetisch grondwaarheidsgewicht per krop
   (gewicht = f(oppervlakte) + ruis). Dit vervangt de ontbrekende echte data.

2. `src/detection/` — YOLO11-seg training/inference wrapper (ultralytics).
   Dataset-loader die COCO-json naar YOLO-seg-formaat converteert. Trainscript
   met configurable epochs/imgsz, default 50 epochs op de synthetische set
   zodat het end-to-end draait op CPU binnen enkele minuten (documenteer hoe
   dit op te schalen met een echte GPU/Colab voor productiegebruik).

3. `src/calibration/gsd.py` — GSD-berekening: cm/pixel = (sensorbreedte_mm *
   vlieghoogte_m) / (brandpuntsafstand_mm * beeldbreedte_px). Functie die
   pixeloppervlakte omzet naar cm² gegeven vluchtparameters.

4. `src/regression/weight_model.py` — features uit segmentatiemasker
   (oppervlakte, omtrek, compactheid = 4π·oppervlakte/omtrek², convex-hull-
   ratio) → Ridge-regressie (scikit-learn) naar geschat versgewicht.
   Classificeer in <350g / 350-500g / >500g. Train + evalueer (R², MAE) op
   de synthetische set; rapporteer metrics naar `results/regression_report.md`.

5. `src/sampling/stats.py` — Cochran's steekproefformule + finite population
   correction + bootstrap-BI (scipy.stats.bootstrap, BCa) om te bepalen welk
   steekproefpercentage (10/25/50%) een betrouwbare schatting geeft van de
   gewichtsklasseverdeling. CLI-tool die dit voor een gegeven populatiegrootte
   en gewenste betrouwbaarheid uitrekent.

6. `src/dashboard/app.py` — Streamlit-dashboard: laadt detectie-CSV (kropID,
   x, y, oppervlakte_cm2, gewichtsklasse), toont (a) aantallen per klasse,
   (b) kleurgecodeerde veldkaart via streamlit-folium/folium (amber <350g,
   groen 350-500g, roestrood >500g — consistent met het "Bed & Bracket"
   kleurenschema uit de projectkennis als dat aanwezig is), (c) histogram
   per gewichtsklasse. Draai op synthetische data als demo.

7. `src/edge/export_tensorrt.py` — script dat het getrainde YOLO-model
   exporteert naar ONNX (`model.export(format='onnx')`) met duidelijke
   TODO/README voor de TensorRT-INT8-conversiestap die alleen op een
   Jetson-toestel met JetPack kan (niet in deze container).

## Kwaliteitseisen
- Elke module heeft minimaal 3 unit tests (`pytest`, map `tests/`).
- Elke module heeft een korte sectie in het hoofdrepo-README.
- Commit na elke afgeronde module met een duidelijke commit message.
- Aan het eind: een `RESULTS.md` met wat werkt, wat gesynthetiseerd is en dus
  met echte data opnieuw getraind/gevalideerd moet worden, en vervolgstappen.
- Log voortgang doorlopend.

Begin nu met stap 1 en werk de lijst zelfstandig af tot het einde.
