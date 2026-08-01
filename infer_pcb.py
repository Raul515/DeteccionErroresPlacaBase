import os
import glob
from ultralytics import YOLO

def run_inference():
    # 1. Buscar pesos entrenados
    possible_paths = [
        os.path.abspath("runs/detect/pcb_defect_yolov10/weights/best.pt"),
    ]
    
    weights_path = None
    for p in possible_paths:
        if os.path.exists(p):
            weights_path = p
            break
            
    if not weights_path:
        raise FileNotFoundError(
            f"[ERROR] No se encontraron los pesos del modelo en ninguna ruta conocida. "
            f"Ejecuta primero: python train_pcb.py"
        )

    print(f"[INFO] Cargando pesos del modelo: {weights_path}")
    model = YOLO(weights_path)

    # 2. Obtener imágenes de validación
    val_images = glob.glob("dataset/val/images/*.jpg")
    if not val_images:
        print("[ERROR] No se encontraron imagenes en dataset/val/images/")
        return

    print(f"[INFO] Ejecutando inferencia sobre {len(val_images)} imagenes de prueba...")
    
    # 3. Realizar predicción con YOLOv10
    _ = model.predict(
        source=val_images[:5],  # Probar con las primeras 5 imágenes
        conf=0.25,              # Umbral de confianza del 25%
        save=True,
        project="runs",
        name="detect/inference_results",
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("[OK] Inferencias completadas con exito!")
    print("[OUTPUT] Las imagenes anotadas con cuadros delimitadores se guardaron en:")
    print("   runs/detect/inference_results/")
    print("=" * 60)

if __name__ == "__main__":
    run_inference()
