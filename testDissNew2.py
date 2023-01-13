import tensorflow as tf
model = tf.keras.models.load_model('my_modelNewData96.hdf5')
import streamlit as st
st.write("""
         # Alzheimers MRI Prediction
         """
         )
st.write("This is a image multi- class classification web app to predict the stage of Alzheimerd in MRI scans")
st.write("!!Disclaimer, this is a tool to help trained radiologists for a second opinion and not for self diagnosing!!")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np


def import_and_predict(image_data, model):
    
        size = (128,128)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(128,128),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)  
        
        return prediction
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Diagnosis = Mild Demented")
    elif np.argmax(prediction) == 1:
        st.write("Diagnosis = Moderate Demented")
    elif np.argmax(prediction) == 2:
        st.write("Diagnosis = Non Demented")
    else:
        st.write("Diagnosis = Very Mild Demented")
    
    st.text("Probability (0: Mild Demented, 1: ModerateDemented, 2: Non Demented, 3: Very Mild Demented ")
    st.write(prediction)
    
