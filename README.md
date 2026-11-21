---
title: Handwritten Digit Recognition
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---

# Handwritten Digit Recognition using CNN

This Streamlit app loads the trained `digit_model.keras` CNN and predicts one handwritten digit (0–9) from an uploaded image.

## Deploy on Hugging Face Spaces

1. Sign in to https://huggingface.co/ and create a new Space.
2. Choose **Docker** as the Space SDK (the app uses Streamlit on port 8501).
3. Upload all files from this folder, including `digit_model.keras`.
4. Add the Dockerfile from this package if the Space asks for it, then wait for the build to finish.
5. Open the Space's App tab. Its public URL will be shown by Hugging Face.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The image preprocessing intentionally follows the notebook: grayscale, resize to 28×28, and divide pixel values by 255. The model was trained on MNIST, so clear, centered digits resembling MNIST work best.
