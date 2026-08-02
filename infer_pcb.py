"""
infer_pcb.py
~~~~~~~~~~~~
Inferencia con el modelo YOLOv10 entrenado sobre imágenes de validación.
Carga la ruta de pesos desde config.json (DIP).
"""

from pathlib import Path

from ultralytics import YOLO

from config_loader import CONFIG, resolve_path


def run_inference() -> None:
    """Ejecuta inferencia con el modelo entrenado sobre imágenes de validación."""
    # 1. Buscar pesos entrenados desde config
    weights_path = resolve_path(CONFIG["weights_path"])

    if not weights_path.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontraron los pesos del modelo en: {weights_path}. "
            f"Ejecuta primero: python train_pcb.py"
        )

    print(f"[INFO] Cargando pesos del modelo: {weights_path}")
    model = YOLO(str(weights_path))

    # 2. Obtener imágenes de validación
    val_images_dir = resolve_path(CONFIG["dataset_path"]) / "val" / "images"
    val_images = sorted(val_images_dir.glob("*.jpg"))

    if not val_images:
        print(f"[ERROR] No se encontraron imagenes en {val_images_dir}")
        return

    print(f"[INFO] Ejecutando inferencia sobre {len(val_images)} imagenes de prueba...")

    # 3. Realizar predicción (primeras 5 imágenes)
    source = [str(p) for p in val_images[:5]]
    _ = model.predict(
        source=source,
        conf=0.25,
        save=True,
        project="runs",
        name="detect/inference_results",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("[OK] Inferencias completadas con exito!")
    print("[OUTPUT] Las imagenes anotadas con cuadros delimitadores se guardaron en:")
    print("   runs/detect/inference_results/")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()
