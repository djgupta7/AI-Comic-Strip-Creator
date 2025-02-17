import os
import torch
import gradio as gr
import requests
from transformers import pipeline
from diffusers import StableDiffusionPipeline

############################################
# 1. Setup and Configuration
############################################

# Set your Hugging Face token
huggingface_token = "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"

# Set your Gemini API key
gemini_api_key = "AIzaSyBoFeZR3NRC_QyFqiXWMp4E2EzZQCiHQVU"

# Load Stable Diffusion (example: runwayml/stable-diffusion-v1-5)
# Make sure you have enough VRAM (ideally 8GB+ GPU)
model_id = "runwayml/stable-diffusion-v1-5"

try:
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=huggingface_token  # if needed
        )
        pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=huggingface_token  # if needed
        )
        pipe.to("cpu")
except:
    # If the model is public, you might not need a token
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id
        )
        pipe.to("cpu")

############################################
# 2. Dialogue Generation with Gemini
############################################

def generate_dialogue(prompt: str) -> str:
    """
    Generates a comic strip dialogue using the Gemini model.
    """
    try:
        headers = {
            "Authorization": f"Bearer {gemini_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.9
        }
        # Note: Replace the endpoint with Gemini's actual endpoint if different.
        response = requests.post("https://api.gemini.ai/v1/generate", headers=headers, json=data)
        response.raise_for_status()
        dialogue = response.json()["choices"][0]["text"].strip()
        return dialogue
    except requests.exceptions.RequestException as e:
        return f"Error generating dialogue: {e}"

############################################
# 3. Image Generation with Stable Diffusion
############################################

def generate_image(description: str):
    """
    Generates a comic-style image based on the input description.
    """
    image = pipe(description, guidance_scale=7.5, num_inference_steps=20).images[0]
    return image

############################################
# 4. Combine Dialogue & Image in a Comic Panel
############################################

def create_comic_panel(user_prompt: str):
    """
    Takes a user prompt (e.g. 'A superhero squirrel steals nuts')
    and generates a short dialogue plus a relevant image.
    """
    dialogue_text = generate_dialogue(user_prompt)
    comic_image = generate_image(user_prompt + ", comic style")
    return dialogue_text, comic_image

############################################
# 5. Gradio Interface
############################################

with gr.Blocks() as demo:
    gr.Markdown("# AI Comic Strip Creator")
    gr.Markdown("Enter a short description of what you want in your comic strip panel.")
    
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Comic Prompt",
            placeholder="e.g. A time-traveling robot meets a medieval knight..."
        )
    
    generate_button = gr.Button("Generate Comic Panel")
    
    dialogue_output = gr.Textbox(label="Generated Dialogue")
    image_output = gr.Image(label="Generated Comic Image")
    
    generate_button.click(
        fn=create_comic_panel,
        inputs=prompt_input,
        outputs=[dialogue_output, image_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True to create a public link