"""
train_pcb.py
~~~~~~~~~~~~
Entrenamiento de YOLOv10 para detección de defectos en placas PCB.
Carga todos los hiperparámetros desde config.json (DRY, DIP).
"""

import torch
from ultralytics import YOLO

from config_loader import CONFIG, resolve_path


def train() -> None:
    """Ejecuta el entrenamiento del modelo con la configuración centralizada."""
    # 1. Verificar soporte de CUDA en GPU
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    print("=" * 60)
    print(f"[INFO] Dispositivo de entrenamiento: {device_name}")
    print(f"[INFO] Aceleracion por GPU (CUDA): {'ACTIVADA' if cuda_available else 'DESACTIVADA'}")
    print("=" * 60)

    # 2. Cargar modelo preentrenado
    model_name = CONFIG["model_base"]
    print(f"[INFO] Cargando modelo base: {model_name}...")
    model = YOLO(model_name)

    # 3. Validar ruta del archivo data.yaml
    yaml_path = resolve_path(CONFIG["dataset_yaml"])
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontro el archivo de configuracion en {yaml_path}. "
            f"Ejecuta primero: python download_dataset.py"
        )

    # 4. Entrenar con hiperparámetros de config.json
    train_cfg = CONFIG["training"]
    print("[INFO] Iniciando entrenamiento con hiperparametros de config.json...")
    _ = model.train(
        data=str(yaml_path),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        amp=train_cfg["amp"],
        workers=train_cfg["workers"],
        device=device,
        project=train_cfg["project"],
        name=train_cfg["name"],
        exist_ok=True,
        save=True,
        plots=True,
    )

    output_dir = resolve_path(f"{train_cfg['project']}/{train_cfg['name']}")
    print("\n" + "=" * 60)
    print("[OK] Entrenamiento completado!")
    print("[OUTPUT] Los pesos optimizados y las graficas se guardaron en:")
    print(f"   {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train()
