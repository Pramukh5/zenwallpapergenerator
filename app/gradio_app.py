"""
Zen Minimal Wallpaper Generator
Web interface using Gradio
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_PATH = "../models/lora_weights/pytorch_lora_weights.safetensors"

# Check if LoRA weights exist
if not os.path.exists(LORA_WEIGHTS_PATH):
    raise FileNotFoundError(
        f"LoRA weights not found at {LORA_WEIGHTS_PATH}\n"
        "Please make sure you've downloaded pytorch_lora_weights.safetensors "
        "to the models/lora_weights/ folder"
    )

# ============================================================================
# LOAD MODEL (This happens once when the app starts)
# ============================================================================

print("🔄 Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
)

# Use faster scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load your trained LoRA weights
print(f"🔄 Loading LoRA weights from {LORA_WEIGHTS_PATH}...")
pipe.load_lora_weights(LORA_WEIGHTS_PATH)

# Move to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

print(f"✅ Model loaded on {device}!")

# ============================================================================
# GENERATION FUNCTION
# ============================================================================

def generate_wallpaper(
    prompt,
    negative_prompt,
    width,
    height,
    num_steps,
    guidance_scale,
    seed,
):
    """Generate wallpaper based on user inputs"""
    
    # Add style reinforcement to prompt
    enhanced_prompt = f"{prompt}, minimalist zen wallpaper, abstract, wabi sabi aesthetic"
    
    # Set seed for reproducibility
    if seed == -1:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    try:
        # Generate image
        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]
        return image
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# ============================================================================
# PRESET PROMPTS
# ============================================================================

EXAMPLE_PROMPTS = [
    "soft beige circle floating above horizon line, muted earth tones, negative space",
    "abstract mountain silhouette at sunset, minimal geometric shapes, calm mood",
    "crescent moon with curved line, cream and gray palette, serene atmosphere",
    "single brushstroke arc, wabi sabi texture, simple composition",
    "overlapping circles in earth tones, balanced asymmetry, peaceful",
    "minimal wave pattern, soft gradient background, meditative",
]

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    with gr.Blocks(title="Zen Wallpaper Generator", theme=gr.themes.Soft()) as app:
        
        gr.Markdown(
            """
            # 🎨 Zen Minimal Wallpaper Generator
            
            Generate beautiful minimalist wallpapers in Japanese zen aesthetic.
            Perfect for desktop backgrounds, phone wallpapers, or digital art.
            """
        )
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                
                gr.Markdown("### 📝 Prompt")
                prompt_input = gr.Textbox(
                    label="Describe your wallpaper",
                    placeholder="soft circle and curved line, beige tones...",
                    lines=3,
                    value="soft beige circle floating above horizon line, muted earth tones, negative space"
                )
                
                # Example prompts
                gr.Markdown("#### Quick Examples:")
                example_buttons = []
                for example in EXAMPLE_PROMPTS:
                    btn = gr.Button(example[:60] + "...", size="sm")
                    example_buttons.append((btn, example))
                
                gr.Markdown("### ⚙️ Advanced Settings")
                
                with gr.Accordion("Generation Settings", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (what to avoid)",
                        value="busy, cluttered, detailed, complex, text, watermark, signature, realistic photo",
                        lines=2
                    )
                    
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=1024,
                            step=64,
                            value=768
                        )
                        height = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=1024,
                            step=64,
                            value=768
                        )
                    
                    num_steps = gr.Slider(
                        label="Steps (more = better quality, slower)",
                        minimum=20,
                        maximum=50,
                        step=5,
                        value=30
                    )
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (how closely to follow prompt)",
                        minimum=5,
                        maximum=15,
                        step=0.5,
                        value=7.5
                    )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                generate_btn = gr.Button("🎨 Generate Wallpaper", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ### 💡 Tips:
                    - Describe shapes, colors, and mood
                    - Use words like: circle, line, arc, mountain, moon, horizon
                    - Mention colors: beige, cream, earth tones, muted
                    - Keep it simple and zen!
                    """
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Wallpaper",
                    type="pil",
                    height=600
                )
                
                gr.Markdown(
                    """
                    **Right-click the image to save it!**
                    
                    Recommended resolutions:
                    - Desktop: 1920x1080 or 2560x1440
                    - Phone: 1080x1920
                    - Tablet: 1536x2048
                    """
                )
        
        # Connect example buttons
        for btn, example_text in example_buttons:
            btn.click(
                fn=lambda x: x,
                inputs=gr.State(example_text),
                outputs=prompt_input
            )
        
        # Connect generate button
        generate_btn.click(
            fn=generate_wallpaper,
            inputs=[
                prompt_input,
                negative_prompt,
                width,
                height,
                num_steps,
                guidance_scale,
                seed,
            ],
            outputs=output_image
        )
        
        gr.Markdown(
            """
            ---
            Made with ❤️ using Stable Diffusion + LoRA fine-tuning
            """
        )
    
    return app

# ============================================================================
# LAUNCH APP
# ============================================================================

if __name__ == "__main__":
    app = create_interface()
    
    print("\n" + "="*50)
    print("🚀 Starting Zen Wallpaper Generator...")
    print("="*50 + "\n")
    
    app.launch(
        share=True,  # Creates public link for 72 hours
        server_name="127.0.0.1",  # Localhost - use this URL in your browser
        server_port=7860,
    )