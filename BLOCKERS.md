# Blockers & omgevingsbeperkingen

Punten die niet volledig in deze container konden worden afgerond, met de
gekozen werkwijze eromheen. Geen enkel punt blokkeerde de rest van de build.

## 1. Geen echte geannoteerde/gewogen dataset
**Status:** verwacht (kern van de opdracht).
De hele pipeline draait op de **synthetische datagenerator** (module 1). Alle
formaten (COCO-json, gewicht-CSV, detectie-CSV) zijn identiek aan wat echte data
zou leveren, zodat aansluiten van echte data geen codewijziging vereist. Zie
`RESULTS.md` voor precies wat opnieuw getraind/gevalideerd moet worden.

## 2. TensorRT INT8-conversie
**Status:** niet uitvoerbaar hier (by design).
INT8-engine-bouw vereist een **Jetson-toestel met JetPack/TensorRT** en een
representatieve echte kalibratieset. Module 7 exporteert wél naar **ONNX** en
documenteert de volledige Jetson-stappen (`src/edge/TENSORRT.md`).

## 3. Headless container mist `libGL.so.1`
**Status:** opgelost.
De standaard `opencv-python` faalt zonder `libGL`. Opgelost door
`opencv-python-headless` te installeren (staat in `requirements.txt`).

## 4. `github.com` niet bereikbaar in de container
**Status:** omzeild.
Het downloaden van `yolo11n-seg.pt` via github.com time-out; Ultralytics haalt
het gewicht via een mirror op, dus training werkt. Op een normale
omgeving/Colab is dit geen issue.

## 5. YOLO-detectiekwaliteit op CPU-demo
**Status:** bekende beperking, afgevangen.
Een korte CPU-training op de kleine synthetische set levert (nog) geen
betrouwbare maskers. De pijplijn valt daarom terug op de
grondwaarheidspolygonen als detectie-stand-in (duidelijk gelogd), zodat de
downstream-modules end-to-end demonstreerbaar blijven. Voor productie: train
langer op een GPU met een groter model en echte data (zie README module 2).
