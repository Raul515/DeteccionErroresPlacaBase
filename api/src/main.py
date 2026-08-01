import os
import io
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import Response, JSONResponse
from ultralytics import YOLO
from ultralytics.engine.results import Results

app = FastAPI(
    title="PCB Defect Detection API",
    description="API MLOps para la deteccion automatizada de defectos en placas de circuitos usando YOLOv10",
    version="1.0.0"
)

# Cargar modelo YOLOv10
WEIGHTS_PATH = os.getenv("MODEL_PATH", "runs/detect/pcb_defect_yolov10/weights/best.pt")

print(f"[INFO API] Intentando cargar pesos del modelo desde: {WEIGHTS_PATH}")
try:
    model = YOLO(WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else "yolov10s.pt")
    print(f"[INFO API] Modelo cargado exitosamente.")
except Exception as e:
    print(f"[WARN API] No se pudieron cargar los pesos locales ({e}). Usando yolov10s base.")
    model = YOLO("yolov10s.pt")

@app.get("/health")
def health_check():
    """Endpoint de estado del servicio y dispositivo hardware."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "status": "ok",
        "gpu_available": cuda_available,
        "device": device_name,
        "model_loaded": WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else "yolov10s.pt",
        "classes": model.names
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.0, le=1.0)):
    """Devuelve las detecciones en formato JSON estructurado (clase, confianza, bounding boxes)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado debe ser una imagen.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    # Inferencia con YOLO
    results = list(model.predict(source=img_np, conf=conf, verbose=False))
    result = results[0]

    detections = []
    if isinstance(result, Results) and result.boxes is not None:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            names = model.names or {}
            class_name = names.get(class_id, f"class_{class_id}")

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": [round(c, 2) for c in coords]
            })

    return JSONResponse(content={
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections
    })

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    show_labels: bool = Query(True),
    show_conf: bool = Query(True)
):
    """Devuelve la imagen renderizada con los cuadros delimitadores dibujados."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado debe ser una imagen.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    # Inferencia con YOLO
    results = list(model.predict(source=img_np, conf=conf, verbose=False))
    result = results[0]

    if not isinstance(result, Results):
        raise HTTPException(status_code=500, detail="Error inesperado en la inferencia del modelo.")

    # Plotear predicciones usando Ultralytics con los parametros personalizados
    annotated_frame = result.plot(labels=show_labels, conf=show_conf)
    annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Convertir frame a bytes JPEG
    is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
    if not is_success:
        raise HTTPException(status_code=500, detail="Error codificando la imagen procesada.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
