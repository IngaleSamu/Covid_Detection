import streamlit as st
import tensorflow as tf
import streamlit as st

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.models import load_model
from keras.preprocessing import image


rad = st.sidebar.radio("Covid Detector", ["Home", "About Us", "Predict"])
if rad=="Home":
    from PIL import Image
    st.header("COVID-19 VIRUS")
    img = Image.open("./gettyimages-1203376093.png")
    st.image(img)
    col1,col2,col3 =st.beta_columns(3)
    nav1 = col1.button('Overview')
    nav2 = col2.button('Prevention')
    nav3 = col3.button('Symptoms')
    if nav1:
        st.write("Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.")
        st.write("Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.")
        st.write("The best way to prevent and slow down transmission is to be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. ")
        st.write("The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow).")
    if nav2:
        st.markdown(""" ## To prevent infection and to slow transmission of COVID-19, do the following:
        """,True)
        st.markdown(""" :mask: _ Wash your hands regularly with soap and water, or clean them with alcohol-based hand rub. _ """, True)
        st.markdown(""" :mask: _ Maintain at least 1 metre distance between you and people coughing or sneezing. _ """, True)
        st.markdown(""" :mask: _ Avoid touching your face. _ """, True)
        st.markdown(""" :mask: _ Cover your mouth and nose when coughing or sneezing. _ """, True)
        st.markdown(""" :mask: _ Stay home if you feel unwell. _ """, True)
        st.markdown(""" :mask: _ Refrain from smoking and other activities that weaken the lungs. _ """, True)
        st.markdown(""" :mask: _ Practice physical distancing by avoiding unnecessary travel and staying away from large groups of people. _ """, True)
        
    if nav3:
        st.markdown(""" ## COVID-19 affects different people in different ways. Most infected people will develop mild to moderate illness and recover without hospitalization. """,True)
        st.markdown(""" ### Most common symptoms: """,True)
        st.markdown(""" :mask: _ Fever. _ """, True)
        st.markdown(""" :mask: _ dry cough. _ """, True)
        st.markdown(""" :mask: _ tiredness. _ """, True)

        st.markdown(""" ### Less common symptoms: """,True)
        st.markdown(""" :mask: _ aches and pains. _ """, True)
        st.markdown(""" :mask: _ sore throat. _ """, True)
        st.markdown(""" :mask: _ diarrhoea. _ """, True)
        st.markdown(""" :mask: _ conjunctivitis. _ """, True)
        st.markdown(""" :mask: _ headache. _ """, True)
        st.markdown(""" :mask: _ loss of taste or smell. _ """, True)
        st.markdown(""" :mask: _ a rash on skin, or discolouration of fingers or toes. _ """, True)

        st.markdown(""" ### Serious symptoms: """,True)
        st.markdown(""" :mask: _ difficulty breathing or shortness of breath. _ """, True)
        st.markdown(""" :mask: _ chest pain or pressure. _ """, True)
        st.markdown(""" :mask: _ loss of speech or movement. _ """, True)
        st.write("Seek immediate medical attention if you have serious symptoms.  Always call before visiting your doctor or health facility. ")
        st.write("People with mild symptoms who are otherwise healthy should manage their symptoms at home.")
        st.write("On average it takes 5–6 days from when someone is infected with the virus for symptoms to show, however it can take up to 14 days. ")
if rad=="About Us":
    st.header("About Us")
    from PIL import Image
    img1 = Image.open("./Coronavirus-e1600864953420.jpg")
    st.image(img1)
    st.markdown(""" ### Covid Detector is a web app, which is helpful to detect if a person is corona positive or healthy by using chest X-ray as a input. This will helpful in hospitals for radiologist to evaluate x-ray and also convinient to people to check their reports are positive or normal.""",True)
    st.markdown(""" ### Covid Detection using X-ray is much more efficient to people because of following aspects : """,True)
    st.markdown(""" #### 1. Blood Tests are Costly. """, True)
    st.markdown(""" #### 2. Blood Tests Consumes more time(approximately 5 hours per patient) rather than X-ray.  """, True)
    st.markdown(""" #### 3. Extent of Spread can also be detected. """, True)
    st.markdown("""  _ The project is built by Samruddhi Ingale! _ """, True)

if rad=="Predict":
    @st.cache(allow_output_mutation=True)
    def load_model():
        model=tf.keras.models.load_model('./model_adv.h5')
        return model
    with st.spinner('Model is being loaded..'):
        model=load_model()

    st.write('Covid Detection')

    file = st.file_uploader("Please upload image of PA view of chest X-ray", type=["jpg", "png",  "jpeg"])
    from PIL import Image,  ImageOps
    from keras.preprocessing import image
    st.set_option('deprecation.showfileUploaderEncoding', False)
    def import_and_predict(image_data, model):
        size = (224,224)    
        dic = {0:'Covid Positive', 1:'Normal'}
        t = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        t = img_to_array(t)
        t = np.expand_dims(t,axis=0)
        result = model.predict(t)        
        return result
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        if predictions==1:
            st.success("The patient is Normal")
        else:
            st.success("The patient is Covid Positive")

        
    