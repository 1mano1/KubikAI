# Guía de Entrenamiento en Kaggle - Kubik AI 2.0

Este documento contiene las rutas exactas, comandos y pasos necesarios para entrenar y depurar los modelos de Kubik AI dentro del entorno de Kaggle.

## Rutas Importantes en Kaggle

*   **Dataset Procesado:** `/kaggle/input/datasets/imanolr11/kubikai-3d-sdf-dataset`
*   **Salida VAE (Versión 2):** `/kaggle/working/vae_training_output_v2`

---

## 🎯 OBJETIVO ACTUAL: Opción A (Refinar VAE)

Has decidido continuar el entrenamiento del VAE funcional (v2) para mejorar su calidad. Sigue estos pasos exactos:

### 1. Preparación del Entorno
Ejecuta esto en la primera celda:

```bash
!git clone -b v3 https://github.com/1mano1/Kubik-AI-2.0.git
%cd Kubik-AI-2.0
!pip install -r requirements.txt
```

### 2. Comando de Reanudación (De 10k a 100k)
Este comando carga tu checkpoint exitoso del paso 10,000 y continúa el entrenamiento.
**Asegúrate de que el directorio `vae_training_output_v2` contenga los checkpoints anteriores** (si es una nueva sesión de Kaggle, necesitarás mover o descargar/subir el checkpoint `vae_step0010000.pt` a esa ruta, o ajustar el `--load_dir`).

Si estás en la **misma sesión** donde entrenaste los primeros 10k pasos:

```bash
!python KubikAI/train_vae.py --config KubikAI/configs/kubikai_sdf_vae_v1.json --output_dir /kaggle/working/vae_training_output_v2 --data_dir /kaggle/input/datasets/imanolr11/kubikai-3d-sdf-dataset --load_dir /kaggle/working/vae_training_output_v2 --resume_step 10000
```

> **Nota Crítica:** Si has reiniciado la sesión de Kaggle, los archivos en `/kaggle/working/` se habrán borrado. Necesitarás volver a subir el archivo `vae_step0010000.pt` a una carpeta, y apuntar `--load_dir` a esa carpeta.

---

## Comandos Estándar (Referencia)

### Iniciar Entrenamiento desde Cero (Nueva Arquitectura)
```bash
!python KubikAI/train_vae.py --config KubikAI/configs/kubikai_sdf_vae_v1.json --output_dir /kaggle/working/vae_training_output_v2 --data_dir /kaggle/input/datasets/imanolr11/kubikai-3d-sdf-dataset
```

### Depuración / Prueba
Reemplaza el número de paso según corresponda:
```bash
!python KubikAI/debug_vae.py --vae_ckpt /kaggle/working/vae_training_output_v2/ckpts/vae_step0050000.pt
```

**Criterio de Éxito:**
*   Busca `Min=-0.xxxx` (Valor negativo).
*   Busca `SUCCESS: Generated a debug shape`.
