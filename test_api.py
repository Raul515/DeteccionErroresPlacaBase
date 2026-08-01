import os
import glob
import json
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def test_pipeline():
    print(f"[INFO TEST] Probando API en: {API_URL}")
    
    # 1. Probar endpoint /health
    try:
        r_health = requests.get(f"{API_URL}/health")
        print("\n" + "=" * 50)
        print("1. [GET /health] Respuesta:")
        print(json.dumps(r_health.json(), indent=2))
        print("=" * 50)
    except Exception as e:
        print(f"[ERROR] No se pudo conectar con la API en {API_URL}: {e}")
        return

    # 2. Obtener una imagen de prueba
    val_images = glob.glob("dataset/val/images/*.jpg")
    if not val_images:
        print("[ERROR] No se encontraron imagenes en dataset/val/images/ para probar.")
        return
    
    test_img_path = val_images[0]
    print(f"\n[INFO TEST] Usando imagen de prueba: {test_img_path}")

    # 3. Probar endpoint /predict (JSON)
    with open(test_img_path, "rb") as f:
        files = {"file": (os.path.basename(test_img_path), f, "image/jpeg")}
        r_pred = requests.post(f"{API_URL}/predict?conf=0.25", files=files)
        print("\n2. [POST /predict] Respuesta JSON:")
        print(json.dumps(r_pred.json(), indent=2))

    # 4. Probar endpoint /detect (Imagen anotada)
    with open(test_img_path, "rb") as f:
        files = {"file": (os.path.basename(test_img_path), f, "image/jpeg")}
        r_det = requests.post(f"{API_URL}/detect?conf=0.25&show_labels=true&show_conf=true", files=files)
        
        output_image_path = "api_detection_result.jpg"
        if r_det.status_code == 200:
            with open(output_image_path, "wb") as out_f:
                out_f.write(r_det.content)
            print(f"\n3. [POST /detect] Imagen anotada guardada en: {os.path.abspath(output_image_path)}")
        else:
            print(f"[ERROR] Error en /detect: {r_det.status_code} - {r_det.text}")

if __name__ == "__main__":
    test_pipeline()
