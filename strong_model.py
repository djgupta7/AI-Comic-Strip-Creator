import torch
import gradio as gr
from transformers import pipeline, set_seed
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

############################################
# 1. Setup and Configuration
############################################

# Set device (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-2 model for dialogue generation
set_seed(42)
text_gen = pipeline("text-generation", model="gpt2", device=0 if device == "cuda" else -1)

# Load Stable Diffusion Model (dreamlike-photoreal-2.0)
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)

############################################
# 2. Dialogue Generation with GPT-2
############################################

def generate_dialogue(prompt: str) -> str:
    """
    Generates a short comic strip dialogue using GPT-2.
    """
    response = text_gen(prompt, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.9)
    return response[0]['generated_text'].strip()

############################################
# 3. Image Generation with Stable Diffusion
############################################

def generate_image(description: str) -> Image:
    """
    Generates a photorealistic comic-style image using Stable Diffusion.
    """
    image = pipe(description, guidance_scale=7.5, num_inference_steps=25).images[0]
    return image

############################################
# 4. Combine Dialogue & Image in a Comic Panel
############################################

def save_comic_panel(dialogue_text: str, image: Image) -> Image:
    """
    Adds dialogue text to an image and returns the final comic panel.
    """
    width, height = image.size
    new_height = height + 100  # Extra space for dialogue

    new_image = Image.new("RGB", (width, new_height), "white")
    new_image.paste(image, (0, 0))

    # Draw text
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()
    text_position = (10, height + 10)
    draw.text(text_position, dialogue_text, fill="black", font=font)

    return new_image

############################################
# 5. Full Comic Panel Generation
############################################

def create_comic_panel(user_prompt: str):
    """
    Generates a full comic panel with both dialogue and an image.
    """
    dialogue_text = generate_dialogue(user_prompt)
    image = generate_image(user_prompt + ", comic style")
    final_panel = save_comic_panel(dialogue_text, image)

    return dialogue_text, image, final_panel

############################################
# 6. Gradio Interactive UI (Enhanced)
############################################

with gr.Blocks(css="body { background-color: #1e1e1e; color: white; font-family: 'Arial', sans-serif; }") as demo:
    gr.HTML("<h1 style='text-align: center; color: #FFCC00;'>🎨 AI Comic Strip Creator</h1>")
    gr.HTML("<p style='text-align: center;'>Create AI-generated comic panels with realistic images & dialogues!</p>")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="✨ Enter Comic Idea", placeholder="e.g. A robot meets a medieval knight...", interactive=True)
            generate_button = gr.Button("🚀 Generate Comic Panel", variant="primary")

        with gr.Column():
            dialogue_output = gr.Textbox(label="💬 AI-Generated Dialogue", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🖼 Generated Image")
            generated_image = gr.Image(label="Generated Image", interactive=False, width=400)

        with gr.Column():
            gr.Markdown("### 🎭 Final Comic Panel")
            final_comic = gr.Image(label="Final Comic Panel", interactive=False, width=400)

    generate_button.click(
        fn=create_comic_panel,
        inputs=prompt_input,
        outputs=[dialogue_output, generated_image, final_comic]
    )

if __name__ == "__main__":
    demo.launch(share=True)
