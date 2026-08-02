"""
config_loader.py
~~~~~~~~~~~~~~~~
Módulo centralizado de configuración del proyecto.
Carga config.json con validación y valores por defecto seguros.
Todas las rutas se resuelven con pathlib relativas a PROJECT_ROOT.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Raíz del proyecto: directorio donde reside este archivo
PROJECT_ROOT = Path(__file__).resolve().parent

_CONFIG_PATH = PROJECT_ROOT / "config.json"

# Valores por defecto — usados si config.json falta o tiene campos ausentes
_DEFAULTS = {
    "dataset_path": "dataset",
    "dataset_yaml": "dataset/data.yaml",
    "model_base": "yolov10s.pt",
    "weights_path": "runs/detect/pcb_defect_yolov10/weights/best.pt",
    "training": {
        "epochs": 25,
        "imgsz": 640,
        "batch": 4,
        "amp": False,
        "workers": 0,
        "project": "runs/detect",
        "name": "pcb_defect_yolov10",
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "docker_port": 8001,
        "max_upload_mb": 10,
    },
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Mezcla recursiva: los valores de overrides sobreescriben defaults."""
    merged = defaults.copy()
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict:
    """Carga config.json y lo mezcla con los defaults.

    Returns:
        dict con la configuración completa y validada.
    """
    if _CONFIG_PATH.exists():
        try:
            raw = _CONFIG_PATH.read_text(encoding="utf-8")
            user_config = json.loads(raw)
            config = _deep_merge(_DEFAULTS, user_config)
            logger.info("Configuración cargada desde %s", _CONFIG_PATH)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Error leyendo %s (%s). Usando valores por defecto.", _CONFIG_PATH, exc
            )
            config = _DEFAULTS.copy()
    else:
        logger.warning(
            "%s no encontrado. Usando valores por defecto.", _CONFIG_PATH
        )
        config = _DEFAULTS.copy()

    return config


def resolve_path(relative: str) -> Path:
    """Resuelve una ruta relativa al PROJECT_ROOT.

    Args:
        relative: Ruta relativa como string (e.g. 'dataset/data.yaml').

    Returns:
        pathlib.Path absoluta.
    """
    return (PROJECT_ROOT / relative).resolve()


# Configuración singleton cargada al importar el módulo
CONFIG = load_config()
