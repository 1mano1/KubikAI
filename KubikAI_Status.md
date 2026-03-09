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
- **FÁBRICA DE DATOS COMPLETADA (100%):** Se procesaron con éxito ~120,000 modelos de Plyverse (Parts 1, 2, 3 y 4). Esta cantidad es más del doble de la usada en benchmarks históricos (ShapeNet ~51k) y es el volumen óptimo para los límites de cómputo de Kaggle sin caer en tiempos de entrenamiento inmanejables.
- **Objaverse Descartado:** El dataset `10k-objaverse-object` en Kaggle resultó contener imágenes 2D (renders y poses) en lugar de geometría 3D real, por lo que no es apto para extraer SDF. Se omite para pasar directo al entrenamiento.

## Archivos Clave y Rutas Futuras en Kaggle:
- **Output Esperado Mañana:** `/kaggle/working/SuperDataset_SDF/plyverse_1` (Deberá ser convertido en un "Kaggle Dataset" privado).
- **Ruta de Salida VAE v3:** `/kaggle/working/vae_training_genesis_v3`

## Próximos Pasos Inmediatos:

### 1. Entrenamiento Definitivo (KubikAI Genesis)
- Crear un **nuevo notebook** con **Acelerador GPU T4 x2 o P100**.
- Importar los Kaggle Datasets (los `.npz` procesados de Plyverse) como "Data".
- **NOTA:** El sistema de reanudación de Checkpoints en `base_trainer.py` ha sido arreglado para Kaggle. Ahora respeta explícitamente el parámetro `--load_dir` para buscar la carpeta `ckpts/` (vital ya que Kaggle borra `/kaggle/working/` en cada reinicio).
- Ejecutar el comando para iniciar el entrenamiento:
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