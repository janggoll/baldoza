import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow import keras
from tqdm import tqdm

MODEL_URL = "replace_with_huggingface_url"
MODEL_PATH = "model.h5"
model_downloaded = False

def download_model(model_url, model_path):
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(model_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def predict_image(model_path, image):
    # Load the model
    model = keras.models.load_model(model_path)

    # Preprocess the image
    image = image.resize((256, 256))  # Resize the image to match the model's input shape
    image = np.array(image) / 255.0   # Normalize the image pixel values
    image = np.expand_dims(image, axis=0)  # Add an extra dimension to match the model's input shape

    # Make prediction
    prediction = model.predict(image)
    predicted_class = "Dog" if prediction[0][0] > 0.5 else "Cat"

    return predicted_class


def main():
    global model_downloaded
    st.title("Dog vs Cat Image Classification")
    st.write("Upload an image and the app will predict whether it contains a dog or a cat.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to a PIL Image object
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if not model_downloaded:
            # Download the model file
            with st.spinner('Downloading model...'):
                download_model(MODEL_URL, MODEL_PATH)
                model_downloaded = True
            st.success('Model downloaded successfully!')

        # Make prediction on the uploaded image
        predicted_class = predict_image(MODEL_PATH, image)

        # Display the prediction
        st.write("Prediction:", predicted_class)


if __name__ == "__main__":
    main()
