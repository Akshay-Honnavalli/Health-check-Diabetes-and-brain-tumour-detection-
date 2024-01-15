import pickle
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cv2
from PIL import Image


# loading the saved models

heart_disease_model = joblib.load(open("heart_model.sav",'rb'))
brain_tumor_model=joblib.load(open("Hrishi_brain_model.sav",'rb'))

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Heart Attack Estimator',
                           'Brain Tumor Classifier'],
                          icons=['activity','brain','person'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Heart Attack Estimator'):




# Heart Disease Prediction Page

    
    # page title
        st.title('Heart Disease Prediction using ML')
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
                age = st.number_input('Age')
        
        with col2:
                sex = st.number_input('Sex')
        
        with col3:
                cp = st.number_input('Chest Pain types')
        
        with col1:
                trestbps = st.number_input('Resting Blood Pressure')
        
        with col2:
                chol = st.number_input('Serum Cholestoral in mg/dl')
        
        with col3:
                fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
        with col1:
                restecg = st.number_input('Resting Electrocardiographic results')
        
        with col2:
                thalach = st.number_input('Maximum Heart Rate achieved')
        
        with col3:
                exang = st.number_input('Exercise Induced Angina')
        
        with col1:
                oldpeak = st.number_input('ST depression induced by exercise')
        
        with col2:
                slope = st.number_input('Slope of the peak exercise ST segment')
        
        with col3:
                ca = st.number_input('Major vessels colored by flourosopy')
        
        with col1:
                thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
        heart_diagnosis = ''
    
    # creating a button for Prediction
    
        if st.button('Heart Disease Test Result'):
                heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
                if (heart_prediction[0]==1):
                        heart_diagnosis = 'The person is having heart disease'
                else:
                        heart_diagnosis = 'The person does not have any heart disease'
        
        st.success(heart_diagnosis)
              
elif(selected=='Brain Tumor Classifier'):
        # st.title('BRAIN TUMOUR')
        st.title('Brain Tumor Classifier using ML')
        image = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
        if image is not None:
                image=Image.open(image)
                st.image(image, caption='Uploaded Image', use_column_width=True)
        brain=''
        if(st.button('Brain Tumor Result')):
                #img=cv2.imread(image)
                
                img = image.resize((150,150))
                img_array = np.array(img)
                # img_array = img_array.reshape(1,150,150,3)
                a=brain_tumor_model.predict(img_array)
                indices = a.argmax()
                

                if(indices==0):
                        brain='The tumor is Glioma Tumor'
                elif(indices==1):
                        brain='The tumor is Meningioma Tumor'
                elif(indices==2):
                        brain='No tumor'
                else:
                        brain='Pituitary tumor'
                #print(labels[indices])
        st.success(brain)
