"""
api/src/main.py
~~~~~~~~~~~~~~~
API MLOps para detección de defectos en placas PCB con YOLOv10.

Mejoras de seguridad y robustez aplicadas:
  - Streaming chunked upload: previene ataques DoS por agotamiento de memoria.
  - Validación de imágenes: captura UnidentifiedImageError y DecompressionBombError devolviendo 400.
  - CORS Middleware: integración estándar para aplicaciones frontend.
  - Manejo defensivo de inferencia y hardware (CUDA).
  - Integración con config_loader y variables de entorno (DIP).
"""

import io
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
# Configuración Centralizada / Fallback seguro
# ──────────────────────────────────────────────────────────────────────

try:
    from config_loader import CONFIG, resolve_path
    _default_weights = resolve_path(CONFIG.get("weights_path", "runs/detect/pcb_defect_yolov10/weights/best.pt"))
    _default_max_mb = CONFIG.get("api", {}).get("max_upload_mb", 10)
except ImportError:
    _default_weights = Path("runs/detect/pcb_defect_yolov10/weights/best.pt")
    _default_max_mb = 10

WEIGHTS_PATH = Path(os.getenv("MODEL_PATH", str(_default_weights)))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", str(_default_max_mb)))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB por bloque para lectura segura en streaming

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
# App FastAPI y Middlewares
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PCB Defect Detection API",
    description=(
        "API MLOps para la detección automatizada de defectos "
        "en placas de circuitos impresos usando YOLOv10"
    ),
    version="1.0.0",
)

# Soporte Cross-Origin Resource Sharing (CORS) para clientes web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Helpers de Validación y Ejecución
# ──────────────────────────────────────────────────────────────────────

async def _read_and_validate_image(file: UploadFile) -> np.ndarray:
    """Lee en streaming, valida tipo y tamaño, y decodifica la imagen subida.

    Args:
        file: Archivo subido por el usuario.

    Returns:
        Imagen como ndarray RGB.

    Raises:
        HTTPException: 400 si el tipo no es imagen o el archivo está corrupto.
        HTTPException: 413 si excede el tamaño máximo permitido.
    """
    # 1. Validar Content-Type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo enviado debe ser una imagen valida (image/*).",
        )

    # 2. Validar tamaño por header si está disponible
    if file.size and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"La imagen excede el limite maximo de {MAX_UPLOAD_MB} MB.",
        )

    # 3. Leer contenido en bloques para prevenir agotamiento de memoria (DoS)
    contents = bytearray()
    while chunk := await file.read(UPLOAD_CHUNK_SIZE):
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"La imagen excede el limite maximo de {MAX_UPLOAD_MB} MB.",
            )

    # 4. Decodificar la imagen de forma segura
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        if img_np.size == 0 or img_np.ndim != 3:
            raise ValueError("Dimensiones de imagen no validas.")
        return img_np
    except (UnidentifiedImageError, Image.DecompressionBombError, ValueError, OSError) as exc:
        logger.warning("Error decodificando imagen '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=400,
            detail="El archivo proporcionado no es una imagen valida o se encuentra corrupto.",
        )


def _run_inference(img_np: np.ndarray, conf: float) -> Results:
    """Ejecuta inferencia YOLO sobre una imagen numpy con manejo de errores.

    Args:
        img_np: Imagen como ndarray RGB.
        conf: Umbral de confianza.

    Returns:
        Objeto Results de Ultralytics.

    Raises:
        HTTPException: Si la inferencia produce un fallo interno.
    """
    try:
        results = list(model.predict(source=img_np, conf=conf, verbose=False))
        if not results or not isinstance(results[0], Results):
            raise HTTPException(
                status_code=500,
                detail="Error inesperado en la inferencia del modelo.",
            )
        return results[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error durante la inferencia: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Fallo interno durante la inferencia: {exc}",
        )


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Endpoint de estado del servicio y dispositivo hardware."""
    cuda_available = torch.cuda.is_available()
    try:
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    except Exception:
        device_name = "CUDA (dispositivo no consultable)"

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
    app_import = "api.src.main:app" if Path("api/src/main.py").exists() else "main:app"
    uvicorn.run(app_import, host=host, port=port, reload=True)

