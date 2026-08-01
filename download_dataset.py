import os
import sys
import zipfile
import urllib.request
import numpy as np
import cv2

DATASET_DIR = "dataset"
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

def create_sample_pcb_dataset():
    """Crea un dataset de muestra de PCB totalmente funcional con imágenes y etiquetas formato YOLO para pruebas inmediatas."""
    print("[INFO] Configurando dataset de PCB con 6 clases de defectos...")
    
    classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
    
    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        num_images = 50 if split == "train" else 15
        for i in range(num_images):
            # Simular imagen de placa de circuito impreso (fondo verde con pistas doradas)
            img = np.ones((640, 640, 3), dtype=np.uint8) * np.array([35, 110, 45], dtype=np.uint8)
            
            # Dibujar pistas electrónicas
            for _ in range(10):
                pt1 = (int(np.random.randint(50, 590)), int(np.random.randint(50, 590)))
                pt2 = (int(np.random.randint(50, 590)), int(np.random.randint(50, 590)))
                cv2.line(img, pt1, pt2, (60, 215, 240), thickness=int(np.random.randint(4, 10)))
            
            # Generar entre 1 y 3 defectos por imagen
            num_defects = np.random.randint(1, 4)
            lbl_lines = []
            
            for _ in range(num_defects):
                class_id = int(np.random.randint(0, len(classes)))
                cx, cy = float(np.random.uniform(0.15, 0.85)), float(np.random.uniform(0.15, 0.85))
                w, h = float(np.random.uniform(0.06, 0.14)), float(np.random.uniform(0.06, 0.14))
                
                px_min = int((cx - w/2) * 640)
                py_min = int((cy - h/2) * 640)
                px_max = int((cx + w/2) * 640)
                py_max = int((cy + h/2) * 640)
                
                # Dibujar imperfección simulada
                cv2.rectangle(img, (px_min, py_min), (px_max, py_max), (20, 20, 220), -1)
                lbl_lines.append(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")
            
            img_path = os.path.join(img_dir, f"pcb_{split}_{i:03d}.jpg")
            lbl_path = os.path.join(lbl_dir, f"pcb_{split}_{i:03d}.txt")
            
            cv2.imwrite(img_path, img)
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.writelines(lbl_lines)

    abs_dataset_dir = os.path.abspath(DATASET_DIR).replace("\\", "/")
    yaml_content = f"""path: {abs_dataset_dir}
train: train/images
val: val/images

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""
    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[OK] Dataset preparado exitosamente en: {os.path.abspath(DATASET_DIR)}")

def prepare_dataset():
    if os.path.exists(DATA_YAML_PATH):
        print(f"[OK] El dataset ya esta presente en: {DATASET_DIR}")
        return
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    create_sample_pcb_dataset()

if __name__ == "__main__":
    prepare_dataset()
