"""
tests/test_api_suite.py
~~~~~~~~~~~~~~~~~~~~~~~
Suite de tests unitarios y de integración para la API de detección PCB.
Ejecutable directamente con: python -m unittest tests/test_api_suite.py
"""

import io
import unittest
from pathlib import Path
from PIL import Image

from fastapi.testclient import TestClient
from api.src.main import app, MAX_UPLOAD_MB
from config_loader import CONFIG, resolve_path


class TestPCBDefectAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.val_dir = resolve_path(CONFIG.get("dataset_path", "dataset")) / "val" / "images"
        cls.val_images = sorted(cls.val_dir.glob("*.jpg"))

    def test_health_endpoint(self):
        """El endpoint /health debe retornar 200 con status ok y metadatos."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("gpu_available", data)
        self.assertIn("device", data)
        self.assertIn("classes", data)

    def test_predict_endpoint_with_real_image(self):
        """El endpoint /predict debe retornar 200 y la lista de detecciones."""
        if not self.val_images:
            self.skipTest("No hay imágenes de validación en dataset/")

        test_img = self.val_images[0]
        with open(test_img, "rb") as f:
            response = self.client.post(
                "/predict",
                params={"conf": 0.25},
                files={"file": (test_img.name, f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], test_img.name)
        self.assertIn("total_detections", data)
        self.assertIn("detections", data)
        self.assertIsInstance(data["detections"], list)

    def test_detect_endpoint_returns_jpeg(self):
        """El endpoint /detect debe devolver bytes JPEG con content-type image/jpeg."""
        if not self.val_images:
            self.skipTest("No hay imágenes de validación en dataset/")

        test_img = self.val_images[0]
        with open(test_img, "rb") as f:
            response = self.client.post(
                "/detect",
                params={"conf": 0.25, "show_labels": True, "show_conf": True},
                files={"file": (test_img.name, f, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertGreater(len(response.content), 0)

    def test_reject_non_image_content_type(self):
        """Archivos con Content-Type distinto de image/* deben recibir 400 Bad Request."""
        response = self.client.post(
            "/predict",
            files={"file": ("test.txt", b"plain text", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("imagen valida", response.json()["detail"])

    def test_reject_corrupted_image_file(self):
        """Archivos no válidos enviados como image/jpeg deben recibir 400, no 500."""
        response = self.client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"NOT_A_REAL_JPEG_PAYLOAD", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("no es una imagen valida", response.json()["detail"])

    def test_reject_oversized_payload(self):
        """Archivos que excedan el límite configurado deben recibir 413 Payload Too Large."""
        oversized_bytes = b"0" * ((MAX_UPLOAD_MB + 1) * 1024 * 1024)
        response = self.client.post(
            "/predict",
            files={"file": ("huge.jpg", oversized_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 413)

    def test_invalid_confidence_threshold(self):
        """Umbrales de confianza fuera del rango [0.0, 1.0] deben recibir 422 Unprocessable Entity."""
        if not self.val_images:
            self.skipTest("No hay imágenes de validación en dataset/")

        test_img = self.val_images[0]
        with open(test_img, "rb") as f:
            response = self.client.post(
                "/predict",
                params={"conf": 1.5},
                files={"file": (test_img.name, f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
