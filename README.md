# AI Comic Strip Creator

The AI Comic Strip Creator generates engaging comic panels by combining dialogue generation using the Gemini API and image creation using Stable Diffusion. This project leverages Gradio for the interactive UI, allowing users to input a comic prompt and receive both creative dialogue and a comic-style image.

## Features

- **Dialogue Generation:**  
  Generates short, witty, and engaging comic dialogues using the Gemini API.
- **Image Generation:**  
  Creates comic-style images with Stable Diffusion.
- **Interactive UI:**  
  Powered by Gradio, making it easy to input prompts and view generated outputs.
- **Cross-Platform:**  
  Automatically uses GPU (if available) or falls back to CPU.

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/Ai-comic-creator.git
   cd Ai-comic-creator
   ```

2. **Create a Virtual Environment**

   On Windows:

   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Configure API Keys**

   Open `app.py` and replace the placeholders with your actual API keys:
   - Hugging Face token
   - Gemini API key

## Usage

To launch the application, run:

```sh
python app.py
```

After starting, Gradio will generate a public URL (if `share=True` is set) which you can use to share your AI Comic Strip Creator interface.

## Project Structure

- **app.py:**  
  Contains the main application code, including functions for dialogue and image generation and the Gradio UI.
- **requirements.txt:**  
  Lists all the required Python packages.
- **README.md:**  
  This file.

## Future Enhancements

- Fine-tuning dialogue output to better match comic style.
- Customizable image styles and themes.
- Error logging and enhanced user feedback.
- Integration with social media platforms for easy sharing.


Happy comic creating!