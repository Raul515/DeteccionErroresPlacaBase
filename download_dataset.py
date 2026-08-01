import os
import shutil
import numpy as np
import cv2

DATASET_DIR = os.path.abspath("dataset")
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

def generate_pcb_dataset():
    """Genera un dataset de 200 placas de circuito impreso (PCB) con las 6 clases estándar de defectos."""
    print("[INFO] Generando dataset completo de placas PCB (200 imagenes)...")
    
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        
    classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
    
    np.random.seed(42)  # Reproducibilidad
    
    for split, count in [("train", 160), ("val", 40)]:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for i in range(count):
            # 1. Base de la placa (verde solder mask o azul industrial)
            bg_color = [35 + np.random.randint(-10, 10), 110 + np.random.randint(-20, 20), 45 + np.random.randint(-10, 10)]
            img = np.ones((640, 640, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
            
            # 2. Dibujar cuadricula de fondo y pads de soldadura (puntos dorados)
            for x in range(40, 600, 60):
                for y in range(40, 600, 60):
                    cv2.circle(img, (x, y), 8, (60, 215, 240), -1)
                    cv2.circle(img, (x, y), 4, (15, 15, 15), -1)
            
            # 3. Dibujar pistas de cobre conectando pads
            for _ in range(15):
                x1, y1 = np.random.choice(range(40, 600, 60)), np.random.choice(range(40, 600, 60))
                x2, y2 = np.random.choice(range(40, 600, 60)), np.random.choice(range(40, 600, 60))
                cv2.line(img, (x1, y1), (x2, y2), (60, 215, 240), thickness=int(np.random.randint(4, 8)))
            
            # 4. Generar defectos etiquetados
            num_defects = np.random.randint(1, 4)
            lbl_lines = []
            
            for _ in range(num_defects):
                class_id = int(np.random.randint(0, len(classes)))
                cx, cy = float(np.random.uniform(0.1, 0.9)), float(np.random.uniform(0.1, 0.9))
                w, h = float(np.random.uniform(0.06, 0.12)), float(np.random.uniform(0.06, 0.12))
                
                px_min = int((cx - w/2) * 640)
                py_min = int((cy - h/2) * 640)
                px_max = int((cx + w/2) * 640)
                py_max = int((cy + h/2) * 640)
                
                # Renderizar caracteristicas visuales por tipo de defecto
                if class_id == 0:  # missing_hole (pad sin perforar)
                    cv2.circle(img, (int(cx*640), int(cy*640)), 12, (60, 215, 240), -1)
                elif class_id == 1:  # mouse_bite (muesca en pista)
                    cv2.ellipse(img, (int(cx*640), int(cy*640)), (15, 8), 45, 0, 360, (30, 90, 40), -1)
                elif class_id == 2:  # open_circuit (corte en pista)
                    cv2.rectangle(img, (px_min, py_min), (px_max, py_max), bg_color, -1)
                elif class_id == 3:  # short (cortocircuito / puente no deseado)
                    cv2.line(img, (px_min, py_min), (px_max, py_max), (50, 200, 230), thickness=6)
                elif class_id == 4:  # spur (rebaba)
                    cv2.line(img, (int(cx*640), int(cy*640)), (int(cx*640)+15, int(cy*640)+15), (50, 200, 230), thickness=5)
                else:  # spurious_copper (mancha de cobre aislada)
                    cv2.circle(img, (int(cx*640), int(cy*640)), 10, (50, 200, 230), -1)

                lbl_lines.append(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")
            
            img_path = os.path.join(img_dir, f"pcb_{split}_{i:03d}.jpg")
            lbl_path = os.path.join(lbl_dir, f"pcb_{split}_{i:03d}.txt")
            
            cv2.imwrite(img_path, img)
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.writelines(lbl_lines)

    abs_dataset_dir = DATASET_DIR.replace("\\", "/")
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

    print(f"[OK] Dataset generado exitosamente en: {DATASET_DIR}")

if __name__ == "__main__":
    generate_pcb_dataset()
