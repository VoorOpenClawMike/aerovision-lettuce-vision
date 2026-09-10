# TensorRT INT8 edge deployment

```
TensorRT INT8 conversion — RUN ON THE JETSON, NOT IN THIS CONTAINER
===================================================================
Prerequisites (on the Jetson, e.g. Orin Nano/NX with JetPack 6.x):
  * JetPack with CUDA, cuDNN and TensorRT installed (`dpkg -l | grep nvinfer`).
  * `pip install ultralytics` (matching the training version) + onnx.
  * A small representative calibration set (~100-500 real orthomosaic crops) —
    INT8 needs real data to calibrate activation ranges; synthetic data will
    NOT give production-representative quantisation.

Option A — via Ultralytics (recommended, one call):
    from ultralytics import YOLO
    model = YOLO("best.pt")
    model.export(
        format="engine",      # TensorRT engine
        int8=True,            # INT8 quantisation
        data="data.yaml",     # calibration data (REAL crops)
        imgsz=640,
        device=0,             # the Jetson GPU
        workspace=4,          # GB
    )
    # -> best.engine

Option B — explicit ONNX -> trtexec:
    yolo export model=best.pt format=onnx opset=12 imgsz=640
    /usr/src/tensorrt/bin/trtexec \
        --onnx=best.onnx --int8 --saveEngine=best.engine \
        --calib=<calibration_cache> --workspace=4096

Validate after conversion:
  * Compare mAP of the INT8 engine vs the FP32/ONNX model on a held-out set;
    accept only if the mAP drop is within tolerance (typ. < 1-2 %).
  * Benchmark latency/throughput with `trtexec --loadEngine=best.engine`.

TODO (blocked here): building/validating the .engine requires the Jetson +
JetPack + real calibration data, none of which exist in this container.

```
