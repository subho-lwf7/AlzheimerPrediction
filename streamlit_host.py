import streamlit as st
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

import numpy as np
from keras.preprocessing.image import img_to_array, load_img

model = tf.keras.models.load_model(".\densenet.h5")

def preprocess_image(image_path, target_size=(128, 128)):
    # Load the uploaded image
    uploaded_image = load_img(image_path, color_mode='grayscale', target_size=target_size)

    # Convert the image to a NumPy array
    img_array = img_to_array(uploaded_image)

    # Duplicate the single channel to create three channels
    img_array = np.concatenate([img_array, img_array, img_array], axis=-1)

    # Normalize pixel values to the range [0, 1]
    img_array /= 255.0

    # Expand dimensions to create a batch (required by the model)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def main():
    st.title("Alzheimer's Prediction App")

    # Upload image through Streamlit's file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    map_dict =dic={'Mild Demented': 0, 'Moderate Demented': 1, 'Non Demented': 2, 'Very Mild Demented': 3}
    idc = {k:v for v, k in dic.items()}

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image",width=100)

        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(uploaded_file)

        try:
            # Make predictions using the loaded model
            predict_x = model.predict(preprocessed_image)
            classes_x = np.argmax(predict_x, axis=1)
            proba = round(np.max(predict_x) * 100, 2)

            # Display prediction information
            st.write(f"{proba} % chances are there that the image is {idc[classes_x[0]]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
