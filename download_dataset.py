"""
download_dataset.py
~~~~~~~~~~~~~~~~~~~
Genera un dataset sintético de placas PCB con 6 clases de defectos
para entrenar YOLOv10.

Principios aplicados:
  - SRP: cada función tiene una sola responsabilidad.
  - OCP: dispatch dict permite añadir defectos sin modificar lógica existente.
  - DRY: constantes centralizadas, sin magic numbers.
"""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from config_loader import CONFIG, resolve_path

# ──────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 640
PAD_GRID_START = 40
PAD_GRID_END = 600
PAD_GRID_STEP = 60
PAD_OUTER_RADIUS = 8
PAD_INNER_RADIUS = 4
NUM_TRACES = 15
TRACE_THICKNESS_RANGE = (4, 8)
DEFECTS_PER_IMAGE = (1, 4)
DEFECT_CENTER_RANGE = (0.1, 0.9)
DEFECT_SIZE_RANGE = (0.06, 0.12)
SPLITS = {"train": 160, "val": 40}

COPPER_COLOR = (60, 215, 240)
DARK_CENTER_COLOR = (15, 15, 15)
TRACE_COLOR = (50, 200, 230)

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

DATASET_DIR = resolve_path(CONFIG["dataset_path"])
DATA_YAML_PATH = resolve_path(CONFIG["dataset_yaml"])


# ──────────────────────────────────────────────────────────────────────
# Funciones de renderizado por tipo de defecto (OCP — dispatch dict)
# ──────────────────────────────────────────────────────────────────────

def _render_missing_hole(
    img: np.ndarray, cx_px: int, cy_px: int, **_kwargs
) -> None:
    """Pad sin perforar — círculo de cobre sólido."""
    cv2.circle(img, (cx_px, cy_px), 12, COPPER_COLOR, -1)


def _render_mouse_bite(
    img: np.ndarray, cx_px: int, cy_px: int, **_kwargs
) -> None:
    """Muesca en pista — elipse inclinada."""
    cv2.ellipse(img, (cx_px, cy_px), (15, 8), 45, 0, 360, (30, 90, 40), -1)


def _render_open_circuit(
    img: np.ndarray, bg_color: list, px_min: int, py_min: int,
    px_max: int, py_max: int, **_kwargs
) -> None:
    """Corte en pista — rectángulo del color de fondo."""
    cv2.rectangle(img, (px_min, py_min), (px_max, py_max), bg_color, -1)


def _render_short(
    img: np.ndarray, px_min: int, py_min: int,
    px_max: int, py_max: int, **_kwargs
) -> None:
    """Cortocircuito — puente no deseado."""
    cv2.line(img, (px_min, py_min), (px_max, py_max), TRACE_COLOR, thickness=6)


def _render_spur(
    img: np.ndarray, cx_px: int, cy_px: int, **_kwargs
) -> None:
    """Rebaba — línea corta."""
    cv2.line(
        img, (cx_px, cy_px), (cx_px + 15, cy_px + 15),
        TRACE_COLOR, thickness=5,
    )


def _render_spurious_copper(
    img: np.ndarray, cx_px: int, cy_px: int, **_kwargs
) -> None:
    """Mancha de cobre aislada — círculo pequeño."""
    cv2.circle(img, (cx_px, cy_px), 10, TRACE_COLOR, -1)


# Dispatch dict: class_id → función de renderizado.
# Para añadir un nuevo defecto basta con agregar una entrada aquí (OCP).
DEFECT_RENDERERS: dict[int, Callable[..., None]] = {
    0: _render_missing_hole,
    1: _render_mouse_bite,
    2: _render_open_circuit,
    3: _render_short,
    4: _render_spur,
    5: _render_spurious_copper,
}


# ──────────────────────────────────────────────────────────────────────
# Funciones con responsabilidad única (SRP)
# ──────────────────────────────────────────────────────────────────────

def _random_bg_color(rng: np.random.Generator) -> list[int]:
    """Genera un color de fondo PCB verde con variación aleatoria."""
    return [
        35 + int(rng.integers(-10, 10)),
        110 + int(rng.integers(-20, 20)),
        45 + int(rng.integers(-10, 10)),
    ]


def _draw_pads_and_traces(img: np.ndarray, rng: np.random.Generator) -> None:
    """Dibuja la cuadrícula de pads de soldadura y pistas de cobre."""
    grid_points = range(PAD_GRID_START, PAD_GRID_END, PAD_GRID_STEP)

    # Pads de soldadura
    for x in grid_points:
        for y in grid_points:
            cv2.circle(img, (x, y), PAD_OUTER_RADIUS, COPPER_COLOR, -1)
            cv2.circle(img, (x, y), PAD_INNER_RADIUS, DARK_CENTER_COLOR, -1)

    # Pistas de cobre
    grid_list = list(grid_points)
    for _ in range(NUM_TRACES):
        x1, y1 = int(rng.choice(grid_list)), int(rng.choice(grid_list))
        x2, y2 = int(rng.choice(grid_list)), int(rng.choice(grid_list))
        thickness = int(rng.integers(*TRACE_THICKNESS_RANGE))
        cv2.line(img, (x1, y1), (x2, y2), COPPER_COLOR, thickness=thickness)


def _generate_defects(
    img: np.ndarray, bg_color: list[int], rng: np.random.Generator
) -> list[str]:
    """Genera defectos aleatorios sobre la imagen y devuelve las líneas YOLO."""
    num_defects = int(rng.integers(*DEFECTS_PER_IMAGE))
    label_lines: list[str] = []

    for _ in range(num_defects):
        class_id = int(rng.integers(0, len(CLASSES)))
        cx = float(rng.uniform(*DEFECT_CENTER_RANGE))
        cy = float(rng.uniform(*DEFECT_CENTER_RANGE))
        w = float(rng.uniform(*DEFECT_SIZE_RANGE))
        h = float(rng.uniform(*DEFECT_SIZE_RANGE))

        cx_px = int(cx * IMAGE_SIZE)
        cy_px = int(cy * IMAGE_SIZE)
        px_min = int((cx - w / 2) * IMAGE_SIZE)
        py_min = int((cy - h / 2) * IMAGE_SIZE)
        px_max = int((cx + w / 2) * IMAGE_SIZE)
        py_max = int((cy + h / 2) * IMAGE_SIZE)

        # Dispatch: O(1) lookup en lugar de cadena if/elif
        renderer = DEFECT_RENDERERS[class_id]
        renderer(
            img=img, bg_color=bg_color,
            cx_px=cx_px, cy_px=cy_px,
            px_min=px_min, py_min=py_min,
            px_max=px_max, py_max=py_max,
        )

        label_lines.append(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")

    return label_lines


def _create_pcb_image(rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Crea una imagen PCB sintética con defectos y devuelve (imagen, labels)."""
    bg_color = _random_bg_color(rng)
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * np.array(
        bg_color, dtype=np.uint8
    )
    _draw_pads_and_traces(img, rng)
    label_lines = _generate_defects(img, bg_color, rng)
    return img, label_lines


def _generate_split(split_name: str, count: int, rng: np.random.Generator) -> None:
    """Genera todas las imágenes y labels de un split (train/val)."""
    img_dir = DATASET_DIR / split_name / "images"
    lbl_dir = DATASET_DIR / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        img, label_lines = _create_pcb_image(rng)

        img_path = img_dir / f"pcb_{split_name}_{i:03d}.jpg"
        lbl_path = lbl_dir / f"pcb_{split_name}_{i:03d}.txt"

        cv2.imwrite(str(img_path), img)
        lbl_path.write_text("".join(label_lines), encoding="utf-8")


def _write_data_yaml() -> None:
    """Escribe el archivo data.yaml requerido por YOLO."""
    abs_dataset = str(DATASET_DIR).replace("\\", "/")
    class_names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))

    yaml_content = (
        f"path: {abs_dataset}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"names:\n"
        f"{class_names}\n"
    )
    DATA_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML_PATH.write_text(yaml_content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────────────────────────────

def generate_pcb_dataset() -> None:
    """Genera un dataset de 200 placas PCB con las 6 clases estándar de defectos."""
    import shutil

    total = sum(SPLITS.values())
    print(f"[INFO] Generando dataset completo de placas PCB ({total} imagenes)...")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    rng = np.random.default_rng(seed=42)  # Reproducibilidad

    for split_name, count in SPLITS.items():
        _generate_split(split_name, count, rng)

    _write_data_yaml()

    print(f"[OK] Dataset generado exitosamente en: {DATASET_DIR}")


if __name__ == "__main__":
    generate_pcb_dataset()
