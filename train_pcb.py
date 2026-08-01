import os
import torch
from ultralytics import YOLO

def train():
    # 1. Verificar soporte de CUDA en GPU
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    print("=" * 60)
    print(f"[INFO] Dispositivo de entrenamiento: {device_name}")
    print(f"[INFO] Aceleracion por GPU (CUDA): {'ACTIVADA' if cuda_available else 'DESACTIVADA'}")
    print("=" * 60)

    # 2. Cargar modelo preentrenado YOLOv10s (o YOLOv8s)
    model_name = "yolov10s.pt"
    print(f"[INFO] Cargando modelo base: {model_name}...")
    model = YOLO(model_name)

    # 3. Validar ruta del archivo data.yaml
    yaml_path = os.path.abspath("dataset/data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"[ERROR] No se encontro el archivo de configuracion en {yaml_path}. "
            f"Ejecuta primero: python download_dataset.py"
        )

    # 4. Hiperparametros optimizados para 4 GB VRAM (GTX 1650)
    print("[INFO] Iniciando entrenamiento con hiperparametros optimizados para 4 GB VRAM...")
    _ = model.train(
        data=yaml_path,
        epochs=25,             # Epocas de entrenamiento
        imgsz=640,             # Resolucion de imagen (640x640)
        batch=4,               # Batch reducido: AMP no es estable en GTX 1650, FP32 usa mas VRAM
        amp=False,             # Desactivado: GTX 1650 causa NaN con AMP segun Ultralytics
        workers=0,             # 0 en Windows para evitar bloqueos de multiprocessing
        device=device,         # GPU 0
        project="runs/detect",
        name="pcb_defect_yolov10",
        exist_ok=True,
        save=True,
        plots=True
    )
    
    print("\n" + "=" * 60)
    print("[OK] Entrenamiento completado!")
    print("[OUTPUT] Los pesos optimizados y las graficas se guardaron en:")
    print("   runs/detect/pcb_defect_yolov10/")
    print("=" * 60)

if __name__ == "__main__":
    train()
