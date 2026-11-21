# Handwritten Digit Recognition

A Streamlit app using the supplied trained CNN model (`digit_model.keras`) to classify one handwritten digit from 0 to 9.

## Files
- `app.py` — Streamlit application and image preprocessing
- `digit_model.keras` — trained model
- `requirements.txt` — Python dependencies
- `runtime.txt` — asks Streamlit Community Cloud to use Python 3.12

## Deploy on Streamlit Community Cloud
1. Upload all four files to the root of your GitHub repository (not inside an extra nested folder).
2. In Streamlit Community Cloud, create/redeploy the app from that repository and branch.
3. Set the main file path to `app.py`.
4. Deploy. The `runtime.txt` file selects Python 3.12, which is compatible with TensorFlow CPU 2.19.1.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Image guidance
The model was trained on MNIST, whose digits are white on a black background and roughly centered. The app converts common dark-ink-on-light-paper photos into that polarity, detects and crops the digit, scales it into a 20x20 area, and centers it on a 28x28 black canvas. Use one clearly visible digit, good contrast, and a plain background.

The preprocessing improves the match to MNIST but cannot guarantee perfect predictions for every handwriting style or photograph. The model itself has not been retrained.
