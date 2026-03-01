# Estado de la Sesión - 25/02/2026

## Resumen del Proyecto: Kubik AI 2.0
- **VAE (Geometría):** EN PROCESO (Fase de Refinamiento)
  - **Arquitectura:** v2 (GELU, menos capas, más estable).
  - **Estado Actual:** Entrenamiento pausado por límite de tiempo de Kaggle en el paso **63,900**.
  - **Prueba 63k:** La evaluación con resolución 128 confirmó un `Min SDF` de -0.0525. El modelo ha aprendido la macro-estructura (forma humanoide lograda), pero aún carece de detalles finos (manos, rostro) y presenta "ruido flotante".
  - **Siguiente Objetivo:** Reanudar desde el paso 63,000 para llegar a los 100,000+ pasos y refinar los detalles.
- **Flow Model (Textura/Detalle):** En espera del VAE final.

## Archivos Clave en Kaggle:
- **Ruta de Salida:** `/kaggle/working/vae_training_output_v2`
- **Dataset Checkpoint 63k:** `/kaggle/input/datasets/imanolr11/checkpoint-step63`
- **Dataset Entrenamiento:** `/kaggle/input/datasets/imanolr11/kubikai-training-data/KubikAI_Processed/processed_datasets`

## Próximos Pasos (Hoja de Ruta):

### Fase 1: Validación de Arquitectura (Sesión Actual en Kaggle)
1.  **Evaluación Final (107k):** Se comprobó que el VAE v2 capturaba la macro-estructura pero fallaba en detalles finos (manos, cara, "efecto barro").
2.  **Fase 2.5 (Mejora Arquitectónica):** Se rediseñó por completo el `SdfVAE` (`KubikAI/models/sdf_vae.py`) para utilizar **Point-Voxel Splatting** y **3D Convolutions**.
    - La resolución subió de `16` a `32`.
    - `latent_dim` bajó de `256` a `128` para equilibrar la memoria.
    - Se eliminó el "cuello de botella" del Max Pooling Global que destruía el detalle local.
3.  **Luz Verde Pendiente:** Entrenar este **Nuevo Modelo Mejorado** (v3) desde cero para validar que el efecto "barro" ha desaparecido completamente antes del entrenamiento masivo.

### Fase 2: Construcción del Super Dataset (Kaggle)
1.  **Plyverse Integration:** Utilizar `process_plyverse_batch.py` en un notebook de Kaggle para procesar por lotes (batches) de 10k-20k modelos de Plyverse Part 1-4.
2.  **Validación de Calidad:** Correr `validate_kaggle_datasets.py` sobre ShapeNet y Plyverse para filtrar modelos rotos o de baja calidad.
3.  **Buffering:** Crear un Kaggle Dataset privado con los archivos `.npz` (SDF) ya procesados para el entrenamiento rápido.

### Fase 3: El Entrenamiento Definitivo (KubikAI Genesis)
1.  **Reset a Step 1:** Iniciar el entrenamiento del VAE desde cero utilizando el Super Dataset combinado (125k+ modelos).
2.  **Iluminación de Estudio:** Aplicar el nuevo sistema de 3 puntos en los renders para preparar el camino al Flow Model hiper-realista.
3.  **Escalado:** Monitorear el entrenamiento masivo buscando una convergencia global que abarque desde geometría orgánica (humanos) hasta inorgánica (armas, objetos).

## Notas Técnicas Actualizadas:
- Se implementó iluminación de 3 puntos (Key, Fill, Back) en `preprocess_data.py`.
- Se creó motor de procesamiento paralelo para Plyverse.
- El repositorio ha sido "independizado" y limpiado de historiales previos.
