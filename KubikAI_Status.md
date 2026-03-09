# Estado de la Sesión - 28/02/2026

## Resumen del Proyecto: Kubik AI 2.0
- **VAE (Geometría):** EN PREPARACIÓN PARA FASE "GÉNESIS" (Arquitectura v3)
  - **Arquitectura:** v3 lista (Point-Voxel Splatting, 3D CNNs).
  - **Resolución & Latente:** `resolución: 32`, `latent_dim: 128`.
  - **Correcciones Críticas Aplicadas:** 
    - **Axis Swap Bug:** Se corrigió un error matemático grave en `sdf_vae.py` donde el Decoder leía la dimensión X como Z (y viceversa) debido a `grid_sample`. Ahora reconstruye espacialmente perfecto.
    - **Rama en GitHub:** Todo el trabajo de esta fase está resguardado en la rama `v3`.
- **Flow Model (Textura/Detalle):** En espera del VAE final.

## Estado de Procesamiento del "Super Dataset" (Kaggle)
- **Bloqueo Resuelto:** El script original de procesamiento tardaba 300 horas por problemas con `trimesh.proximity` y modelos rotos. 
- **Solución Implementada:** Se reescribió `process_plyverse_batch.py` para usar `scipy.spatial.cKDTree` y matemáticas de Álgebra Vectorial (Dot Product con Vertex Normals) para obtener el "Pseudo-SDF". El tiempo de procesamiento pasó de 40 segundos a **~0.1 segundos por modelo** conservando el signo (+/-) necesario para la red neuronal.
- **Dependencias Actualizadas:** `python-fcl` y `rtree` añadidos a `requirements.txt` para aceleración extrema de CPU.
- **Ejecución Actual:** Kaggle está corriendo en segundo plano ("Save & Run All" en sesión de CPU) procesando **Plyverse Part 4**. 

## Archivos Clave y Rutas Futuras en Kaggle:
- **Output Esperado Mañana:** `/kaggle/working/SuperDataset_SDF/plyverse_1` (Deberá ser convertido en un "Kaggle Dataset" privado).
- **Ruta de Salida VAE v3:** `/kaggle/working/vae_training_genesis_v3`

## Próximos Pasos Inmediatos:

### 1. Finalizar la "Fábrica de Datos"
- **Completado:** `Plyverse Part 1, 2, 3`.
- **En Proceso:** `Plyverse Part 4`.
- **Siguiente Tarea:** Procesar el dataset **10k Objaverse Object** usando el script ahora compatible con múltiples formatos (`process_plyverse_batch.py` en la rama `v3`).

### 2. Entrenamiento Definitivo (KubikAI Genesis)
- Crear un **nuevo notebook** con **Acelerador GPU T4 x2 o P100**.
- Importar los Kaggle Datasets (los `.npz` procesados de Plyverse) como "Data".
- Ejecutar el comando para iniciar el entrenamiento desde cero:
  ```bash
  !git clone -b v3 https://github.com/1mano1/Kubik-AI-2.0.git
  %cd Kubik-AI-2.0
  !pip install -r requirements.txt
  
  !python KubikAI/train_vae.py \
      --config KubikAI/configs/kubikai_sdf_vae_v1.json \
      --output_dir /kaggle/working/vae_training_genesis_v3 \
      --data_dir /kaggle/input/TU_DATASET_DE_KAGGLE/plyverse_1,/kaggle/input/TU_DATASET_DE_KAGGLE/plyverse_2
  ```

## Notas Técnicas Actualizadas:
- **No se requieren cambios en el VAE:** Gracias al pseudo-SDF vectorial, los datos nuevos son 100% compatibles con el `Dataloader` (con clamp de `[-0.1, 0.1]`) y el `Marching Cubes` (nivel de extracción `0.0`). Todo el ecosistema de Kubik AI sigue intacto.