# 🎨 Zen Wallpaper Generator

A fine-tuned AI model for generating minimalist Japanese zen-inspired wallpapers using Stable Diffusion and LoRA (Low-Rank Adaptation).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-orange.svg)

## 📋 Overview

This project demonstrates advanced machine learning techniques by fine-tuning Stable Diffusion v1.5 using LoRA to generate high-quality minimalist wallpapers. The model specializes in creating abstract, zen-inspired artwork with muted color palettes and clean compositions.

### Key Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning of Stable Diffusion
- **Custom Dataset Processing**: Automated image preprocessing and augmentation pipeline
- **Interactive Web Interface**: User-friendly Gradio application for real-time generation
- **GPU Acceleration**: CUDA-optimized for fast inference
- **Customizable Parameters**: Full control over resolution, steps, guidance scale, and seed

## 🛠️ Technical Stack

- **Deep Learning Framework**: PyTorch 2.1.0
- **Diffusion Model**: Stable Diffusion v1.5 (Hugging Face Diffusers)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) via PEFT
- **Web Framework**: Gradio 4.16.0
- **Acceleration**: Mixed Precision Training (FP16)
- **Dataset Processing**: PIL, OpenCV

## 📁 Project Structure

```
imagemodel/
├── app/
│   ├── gradio_app.py          # Interactive web interface
│   └── api.py                 # REST API endpoints
├── dataset/
│   ├── raw/                   # Original training images
│   ├── processed/             # Preprocessed images (512x512)
│   ├── autoresize.py          # Image preprocessing script
│   └── captions.txt           # Image captions for training
├── models/
│   └── lora_weights/          # Trained LoRA weights
├── notebooks/
│   ├── 01_dataset_prep.ipynb  # Data preparation workflow
│   ├── 02_lora_training.ipynb # Model fine-tuning
│   └── 03_inference_test.ipynb # Testing and validation
├── outputs/
│   └── generated_samples/     # Generated wallpapers
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ VRAM for training, 4GB+ for inference

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/zen-wallpaper-generator.git
   cd zen-wallpaper-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained weights** (if available)
   ```bash
   # Place pytorch_lora_weights.safetensors in models/lora_weights/
   ```

### Usage

#### 1. Dataset Preparation

```bash
cd dataset
python autoresize.py
```

This will:
- Resize all images in `raw/` to 512x512
- Center crop to maintain aspect ratio
- Save processed images to `processed/`

#### 2. Training (Optional)

Open and run the Jupyter notebooks in order:
1. `01_dataset_prep.ipynb` - Prepare and validate dataset
2. `02_lora_training.ipynb` - Fine-tune the model
3. `03_inference_test.ipynb` - Test the trained model

#### 3. Run the Web Application

```bash
cd app
python gradio_app.py
```

Access the interface at:
- **Local**: http://127.0.0.1:7860
- **Public**: Gradio will generate a shareable link

## 💡 Model Details

### Architecture

- **Base Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- **Fine-tuning**: LoRA with rank 8, alpha 16
- **Target Modules**: Cross-attention layers in U-Net
- **Training Resolution**: 512x512
- **Scheduler**: DPM++ 2M Karras

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Batch Size | 4 |
| Training Steps | 1000 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Mixed Precision | FP16 |

## 🎯 Use Cases

- **Desktop Wallpapers**: Generate unique minimalist backgrounds
- **Mobile Wallpapers**: Create zen-inspired phone backgrounds
- **Digital Art**: Produce abstract art for creative projects
- **Design Mockups**: Generate placeholder images for UI/UX designs

## 📊 Results

The model successfully generates:
- High-quality 512x512 to 1024x1024 images
- Consistent minimalist aesthetic
- Muted, harmonious color palettes
- Abstract geometric compositions
- Zen-inspired themes

### Sample Prompts

```python
"soft beige circle floating above horizon line, muted earth tones, negative space"
"abstract mountain silhouette at sunset, minimal geometric shapes, calm mood"
"crescent moon with curved line, cream and gray palette, serene atmosphere"
```

## 🔧 API Reference

### Gradio Interface

```python
generate_wallpaper(
    prompt: str,              # Text description
    negative_prompt: str,     # What to avoid
    width: int,              # Image width (512-1024)
    height: int,             # Image height (512-1024)
    num_steps: int,          # Inference steps (20-50)
    guidance_scale: float,   # Prompt adherence (5-15)
    seed: int                # Random seed (-1 for random)
) -> PIL.Image
```

## 📈 Performance

- **Training Time**: ~2 hours on NVIDIA RTX 3080 (10GB)
- **Inference Time**: ~3-5 seconds per image (30 steps, 768x768)
- **Model Size**: LoRA weights ~3MB (vs 4GB full model)
- **VRAM Usage**: ~6GB during inference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Stable Diffusion**: CompVis, Stability AI, and LAION
- **Diffusers Library**: Hugging Face
- **LoRA**: Microsoft Research
- **Dataset**: Unsplash (various photographers)

## 📧 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🔗 Links

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Gradio Documentation](https://gradio.app/docs)

---

⭐ If you found this project helpful, please consider giving it a star!
