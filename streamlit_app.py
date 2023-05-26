import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

import numpy as np
from PIL import Image
from tensorflow import keras

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
    st.title("Dog vs Cat Image Classification")
    st.write("Upload an image and the app will predict whether it contains a dog or a cat.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to a PIL Image object
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction on the uploaded image
        predicted_class = predict_image('model.h5', image)

        # Display the prediction
        st.write("Prediction:", predicted_class)


if __name__ == "__main__":
    main()
