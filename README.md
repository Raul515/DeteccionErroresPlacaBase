# Detección de Defectos (Pipeline MLOps con YOLOv10)
**Proyecto Original:** Visión por computador para analizar las losetas térmicas de la nave Starship. 

*(Nota: La implementación actual demuestra un pipeline completo de MLOps utilizando un dataset de defectos en placas de circuito impreso (PCB) como prueba de concepto, diseñado para ser fácilmente escalable a cualquier otro dominio de detección de objetos como las losetas de la Starship).*

---

## 🚀 Características Principales
- **Entrenamiento Optimizado:** Pipeline configurado para entrenar el modelo estado del arte **YOLOv10s**, optimizado para consumir poca VRAM (ideal para GPUs locales como la GTX 1650 de 4GB).
- **Dataset Sintético:** Incluye un generador de datos de prueba (`download_dataset.py`) para validar el pipeline sin necesidad de descargar datasets pesados de internet.
- **API REST MLOps:** Despliegue de los modelos entrenados a través de **FastAPI**, listo para integrarse en producción.
- **Docker & GPU:** Contenedor Docker configurado (`Dockerfile.gpu`) para utilizar la aceleración por hardware (CUDA) en inferencia, gestionado mediante `docker-compose`.
- **Configuración Centralizada:** Uso de un único archivo `config.json` para gestionar rutas e hiperparámetros.

---

## 📁 Estructura del Proyecto

```text
DeteccionLosetasStarship/
├── api/
│   ├── src/
│   │   └── main.py          # Código principal de la API FastAPI
│   ├── Dockerfile.gpu       # Dockerfile optimizado para CUDA
│   └── requirements.txt     # Dependencias de la API
├── dataset/                 # Dataset generado (entrenamiento y validación)
├── runs/detect/             # Artefactos y pesos guardados de YOLOv10
├── config.json              # Configuración centralizada del proyecto
├── config_loader.py         # Módulo cargador de configuración con validación y defaults
├── docker-compose.yml       # Orquestador para levantar la API
├── download_dataset.py      # Script generador de dataset de prueba
├── infer_pcb.py             # Script de validación/inferencia local
├── train_pcb.py             # Script de entrenamiento YOLOv10
├── test_api.py              # Script cliente para probar la API
└── requirements.txt         # Dependencias globales del proyecto local
```

---

## 🛠️ Buenas Prácticas y Refactorización (SOLID, Seguridad y Eficiencia)
El proyecto ha sido refactorizado aplicando patrones modernos:
- **SOLID (SRP, OCP, DIP)**: El generador de dataset usa un *dispatch dict* extensible sin modificar la lógica (Open/Closed), y funciones modulares de responsabilidad única.
- **Seguridad en la API**: Validación estricta del tipo y límite de tamaño de archivo (máximo 10 MB configurable) para evitar denegación de servicio (DoS) por memoria.
- **Eficiencia**: Reemplazo de lógica condicional `if/elif` anidada por mapeos `O(1)`, e incorporación de `pathlib` para asegurar la portabilidad multiplataforma.


---

## ⚙️ Requisitos Previos e Instalación Local

1. **Python 3.11+**
2. **NVIDIA GPU** con soporte CUDA (recomendado para entrenamiento).
3. **Docker** y **NVIDIA Container Toolkit** (si se desea usar Docker).

### Creación del Entorno Virtual (Local)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🛠️ Flujo de Trabajo (Workflow)

El proyecto está diseñado para seguir un flujo MLOps tradicional:

### 1. Generación de Datos
Genera un dataset sintético de prueba (200 imágenes de PCB con defectos etiquetados) para validar que el entorno está correctamente configurado:
```bash
python download_dataset.py
```

### 2. Entrenamiento del Modelo
Entrena YOLOv10 utilizando la configuración centralizada. Automáticamente detectará si existe hardware CUDA disponible y optimizará el batch size (ej: batch=4, sin AMP para GTX 1650).
```bash
python train_pcb.py
```
> Los pesos entrenados (`best.pt` y `last.pt`) y las gráficas de rendimiento se guardarán automáticamente en `runs/detect/pcb_defect_yolov10/`.

### 3. Inferencia Local
Valida el modelo recién entrenado realizando predicciones sobre el subconjunto de validación del dataset:
```bash
python infer_pcb.py
```

---

## 🐳 Despliegue con Docker (API)

Una vez que el modelo está entrenado (existe un archivo `best.pt`), el proyecto se puede servir fácilmente a través de una API REST contenerizada, exponiéndola en el puerto **8001**.

```bash
# Levantar el servicio
docker compose up -d --build

# Ver los logs del servicio
docker compose logs -f pcb-detector-api
```

---

## 📡 API Endpoints

Una vez que el contenedor Docker o FastAPI está corriendo, expone los siguientes endpoints:

- **`GET /health`**
  Verifica la disponibilidad del servidor y si se detectó la aceleración por hardware (GPU). Además, detalla el modelo de pesos cargado y las clases detectables.

- **`POST /predict`**
  Recibe una imagen (formato `multipart/form-data`) y un umbral de confianza (`conf`), devolviendo un **JSON** con la posición exacta (Bounding Boxes), la confianza y el nombre de cada defecto encontrado.

- **`POST /detect`**
  Igual que `/predict`, pero en lugar de JSON devuelve la **imagen renderizada** (JPEG) con los cuadros delimitadores, etiquetas y puntajes de confianza pintados directamente sobre ella. Es ideal para validación visual rápida.

> **Prueba automatizada:** Puedes probar todos los endpoints corriendo el script `python test_api.py` desde el entorno host.

---

## 🎯 Conclusión y Escalabilidad

El objetivo de este proyecto ha sido establecer una base sólida de **MLOps** que abarque todas las fases del ciclo de vida del machine learning: desde la ingesta de datos hasta el servicio en producción.

**Próximos pasos (Adaptación a Starship):**
Dado que la arquitectura (YOLOv10 + FastAPI + Docker CUDA) es totalmente agnóstica al tipo de objetos, para cambiar de detectar "Defectos en PCB" a "Análisis de Losetas de Starship", el flujo de trabajo es el siguiente:
1. Reemplazar la carpeta `dataset/` por las imágenes reales de la Starship anotadas en formato YOLO.
2. Actualizar las clases en `config.json` y `dataset/data.yaml` (ej. "loseta_sana", "loseta_fracturada", "loseta_faltante").
3. Ejecutar nuevamente `python train_pcb.py`.
4. El contenedor en Docker leerá automáticamente los nuevos pesos reiniciándose (gracias al mapeo de volúmenes), sin necesidad de tocar una sola línea de código en la API ni recompilar la imagen de Docker.
