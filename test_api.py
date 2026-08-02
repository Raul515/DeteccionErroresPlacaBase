"""
test_api.py
~~~~~~~~~~~
Tests de integración para la API de detección de defectos PCB.
Todas las llamadas HTTP tienen timeout para evitar bloqueos.
"""

import json
import os
from pathlib import Path

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30  # segundos


def test_pipeline() -> None:
    """Ejecuta la suite de pruebas contra la API."""
    print(f"[INFO TEST] Probando API en: {API_URL}")

    # 1. Probar endpoint /health
    try:
        r_health = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        r_health.raise_for_status()
        print("\n" + "=" * 50)
        print("1. [GET /health] Respuesta:")
        print(json.dumps(r_health.json(), indent=2))
        print("=" * 50)
    except requests.ConnectionError as exc:
        print(f"[ERROR] No se pudo conectar con la API en {API_URL}: {exc}")
        return
    except requests.Timeout:
        print(f"[ERROR] Timeout al conectar con {API_URL}")
        return
    except requests.HTTPError as exc:
        print(f"[ERROR] Respuesta inesperada de /health: {exc}")
        return

    # 2. Obtener una imagen de prueba
    val_images_dir = Path("dataset/val/images")
    val_images = sorted(val_images_dir.glob("*.jpg"))

    if not val_images:
        print(f"[ERROR] No se encontraron imagenes en {val_images_dir} para probar.")
        return

    test_img_path = val_images[0]
    print(f"\n[INFO TEST] Usando imagen de prueba: {test_img_path}")

    # 3. Probar endpoint /predict (JSON)
    with open(test_img_path, "rb") as f:
        files = {"file": (test_img_path.name, f, "image/jpeg")}
        r_pred = requests.post(
            f"{API_URL}/predict",
            params={"conf": 0.25},
            files=files,
            timeout=REQUEST_TIMEOUT,
        )
        r_pred.raise_for_status()
        print("\n2. [POST /predict] Respuesta JSON:")
        print(json.dumps(r_pred.json(), indent=2))

    # 4. Probar endpoint /detect (Imagen anotada)
    with open(test_img_path, "rb") as f:
        files = {"file": (test_img_path.name, f, "image/jpeg")}
        r_det = requests.post(
            f"{API_URL}/detect",
            params={"conf": 0.25, "show_labels": True, "show_conf": True},
            files=files,
            timeout=REQUEST_TIMEOUT,
        )

        output_image_path = Path("api_detection_result.jpg")
        if r_det.status_code == 200:
            output_image_path.write_bytes(r_det.content)
            print(
                f"\n3. [POST /detect] Imagen anotada guardada en: "
                f"{output_image_path.resolve()}"
            )
        else:
            print(f"[ERROR] Error en /detect: {r_det.status_code} - {r_det.text}")


if __name__ == "__main__":
    test_pipeline()
