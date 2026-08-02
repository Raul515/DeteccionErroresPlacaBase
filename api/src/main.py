"""
api/src/main.py
~~~~~~~~~~~~~~~
API MLOps para detección de defectos en placas PCB con YOLOv10.

Mejoras aplicadas:
  - DRY: helpers compartidos para lectura de imagen e inferencia.
  - Seguridad: límite de tamaño de upload configurable.
  - Logging: reemplaza print() por logging estándar.
  - Pathlib: rutas modernas y portables.
"""

import io
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pcb_api")

# ──────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────

# Dentro del contenedor Docker no existe config_loader en la raíz,
# así que leemos directamente de variables de entorno con fallbacks seguros.
WEIGHTS_PATH = Path(
    os.getenv("MODEL_PATH", "runs/detect/pcb_defect_yolov10/weights/best.pt")
)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

# ──────────────────────────────────────────────────────────────────────
# Carga del modelo
# ──────────────────────────────────────────────────────────────────────

FALLBACK_MODEL = "yolov10s.pt"


def _load_model() -> YOLO:
    """Carga el modelo YOLO con fallback al modelo base."""
    if WEIGHTS_PATH.exists():
        target = str(WEIGHTS_PATH)
        logger.info("Cargando pesos entrenados desde: %s", target)
    else:
        target = FALLBACK_MODEL
        logger.warning(
            "Pesos no encontrados en %s. Usando modelo base: %s",
            WEIGHTS_PATH, FALLBACK_MODEL,
        )

    try:
        loaded_model = YOLO(target)
        logger.info("Modelo cargado exitosamente.")
        return loaded_model
    except Exception as exc:
        logger.error("Error cargando modelo %s: %s. Usando fallback.", target, exc)
        return YOLO(FALLBACK_MODEL)


model = _load_model()

# ──────────────────────────────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PCB Defect Detection API",
    description=(
        "API MLOps para la deteccion automatizada de defectos "
        "en placas de circuitos usando YOLOv10"
    ),
    version="1.0.0",
)


# ──────────────────────────────────────────────────────────────────────
# Helpers DRY — lógica compartida entre endpoints
# ──────────────────────────────────────────────────────────────────────

async def _read_and_validate_image(file: UploadFile) -> np.ndarray:
    """Lee, valida tipo y tamaño, y decodifica la imagen subida.

    Args:
        file: Archivo subido por el usuario.

    Returns:
        Imagen como ndarray RGB.

    Raises:
        HTTPException: Si el tipo no es imagen o excede el tamaño máximo.
    """
    # Validar content-type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo enviado debe ser una imagen (image/*).",
        )

    # Leer contenido y validar tamaño
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"La imagen excede el limite de {max_mb:.0f} MB.",
        )

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(image)


def _run_inference(img_np: np.ndarray, conf: float) -> Results:
    """Ejecuta inferencia YOLO sobre una imagen numpy.

    Args:
        img_np: Imagen como ndarray RGB.
        conf: Umbral de confianza.

    Returns:
        Objeto Results de Ultralytics.

    Raises:
        HTTPException: Si la inferencia produce un resultado inesperado.
    """
    results = list(model.predict(source=img_np, conf=conf, verbose=False))
    result = results[0]

    if not isinstance(result, Results):
        raise HTTPException(
            status_code=500,
            detail="Error inesperado en la inferencia del modelo.",
        )

    return result


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Endpoint de estado del servicio y dispositivo hardware."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    loaded_path = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else FALLBACK_MODEL

    return {
        "status": "ok",
        "gpu_available": cuda_available,
        "device": device_name,
        "model_loaded": loaded_path,
        "classes": model.names,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
):
    """Devuelve las detecciones en formato JSON estructurado."""
    img_np = await _read_and_validate_image(file)
    result = _run_inference(img_np, conf)

    detections = []
    if result.boxes is not None:
        names = model.names or {}
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"class_{class_id}")

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": [round(c, 2) for c in coords],
            })

    return JSONResponse(content={
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections,
    })


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    show_labels: bool = Query(True),
    show_conf: bool = Query(True),
):
    """Devuelve la imagen renderizada con los cuadros delimitadores dibujados."""
    img_np = await _read_and_validate_image(file)
    result = _run_inference(img_np, conf)

    # Plotear predicciones con parámetros personalizados
    annotated_frame = result.plot(labels=show_labels, conf=show_conf)
    annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Codificar frame a bytes JPEG
    is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
    if not is_success:
        raise HTTPException(
            status_code=500,
            detail="Error codificando la imagen procesada.",
        )

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
