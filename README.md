# ![KubikAI Logo](KubikAI/logo/KubikAI-Logo.png) KubikAI

KubikAI is an advanced 3D generation AI designed for high-fidelity geometry and texture synthesis. This project is a complete evolution, utilizing a new architecture based on Signed Distance Functions (SDF) and Cross-Attention Flow models to achieve superior results.

## Key Features

- **SDF-VAE Architecture:** Uses Signed Distance Functions for sharp, precise 3D geometries with a 16x16x16 latent grid.
- **Cross-Attention Flow Model:** Implements advanced attention mechanisms for high-fidelity detail synthesis from input images, optimized with DINOv2.
- **Independent Pipeline:** Fully autonomous data processing and training pipeline.
- **Portable & Robust:** Optimized for high-performance environments like Kaggle with reduced memory footprint.

## Project Structure

- `KubikAI/`: Core package containing models, datasets, and trainers.
  - `IMG-TEST/`: Contains test images for inference.
  - `models/`: SDF-VAE and Cross-Attention Flow implementations.
  - `datasets/`: SDF and Latent dataset handlers.
  - `trainers/`: Specialized training logic for each model stage.
  - `configs/`: Training and model configurations.
- `train_*.py`: Entry points for training different model stages.
- `requirements.txt`: Project dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Running Inference on Kaggle

This project is optimized for execution on Kaggle.

1.  **Prepare Datasets:**
    -   Upload your trained VAE checkpoint (`vae_step0050000.pt`) to a Kaggle Dataset.
    -   Upload your trained Flow Model checkpoint (`flow_step0045000.pt`) to another Kaggle Dataset.

2.  **Setup Notebook:**
    -   Create a new Kaggle Notebook.
    -   Add your two checkpoint datasets as input.
    -   Clone this repository: `!git clone https://github.com/1mano1/KubikAI.git`
    -   Navigate into the repo: `%cd Kubik-AI-2.0`
    -   Install dependencies: `!pip install -r requirements.txt`

3.  **Run Inference:**
    -   The final command will depend on the exact paths assigned by Kaggle to your datasets. After discovering the paths using `!ls /kaggle/input/`, the command should look like this:

    ```bash
    !python KubikAI/inference.py \
    --image KubikAI/IMG-TEST/test.jpg \
    --output /kaggle/working/generated_model.obj \
    --vae_ckpt /kaggle/input/datasets/imanolr11/kubikai-weights/vae_step0050000.pt \
    --flow_ckpt /kaggle/input/datasets/imanolr11/kubikai-flow-weights/flow_step0045000.pt \
    --config KubikAI/configs/kubikai_flow_v1.json
    ```

## Credits and Acknowledgments

KubikAI is an independent project developed by **1mano1**, focused on pushing the boundaries of 3D generative AI. 

While the core architecture is inspired by the pioneering work of the **TRELLIS** team, KubikAI introduces significant improvements in stability, data processing, and model efficiency. We acknowledge their research as a foundational pillar for this new, evolved implementation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
