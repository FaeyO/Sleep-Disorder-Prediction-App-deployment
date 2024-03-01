import pickle
import streamlit as st
import sys
from pathlib import Path
import pandas as pd


dir = Path(__file__).resolve().parent.parent
sys.path.append(str(dir))

model = pickle.load(open('./logistic_model.pkl','rb'))

def main():
    st.write("""
    # A simple Sleep Disorder Prediction App
             
    This App Predicts the **Sleep Disorder** of a Patient!            
    """)

    st.sidebar.header('User Input Parameters')

    def user_input_features():
        Age = st.sidebar.slider('Age', 18, 85, 35)
        Sleep_Duration = st.sidebar.slider('Sleep Duration', 4, 5, 9)
        Quality_of_Sleep = st.sidebar.slider('Quality of Sleep', 1, 5, 10)
        Physical_Activity_Level = st.sidebar.slider('Physical Activity Level', 30, 50, 100)
        Stress_Level = st.sidebar.slider('Stress Level', 1, 3, 8)
        Heart_Rate = st.sidebar.slider('Heart Rate', 55, 72, 90)
        Daily_Steps	= st.sidebar.slider('Daily Steps', 1000, 3000, 10000)
        Systolic = st.sidebar.slider('Systolic(mmHg)',90, 120, 180)
        Diastolic = st.sidebar.slider('Diastolic(mmHg)' ,50, 70, 100)
        Gender_Female = st.sidebar.number_input('Female (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Gender_Male  = st.sidebar.number_input('Male (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Accountant = st.sidebar.number_input('Occupation Accountant (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Doctor = st.sidebar.number_input('Occupation Doctor (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Engineer = st.sidebar.number_input('Occupation Engineer (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Lawyer = st.sidebar.number_input('Occupation Lawyer (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Manager  = st.sidebar.number_input('Occupation Manager (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Nurse = st.sidebar.number_input('Occupation Nurse (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Sales_Representative = st.sidebar.number_input('Occupation Sales Rep (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Salesperson = st.sidebar.number_input('Occupation Salesperson (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Scientist = st.sidebar.number_input('Occupation Scientist (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Software_Engineer = st.sidebar.number_input('Occupation Software Engineer (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        Occupation_Teacher = st.sidebar.number_input('Occupation Teacher (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        BMI_Category_Normal = st.sidebar.number_input('BMI-Normal (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        BMI_Category_Obese = st.sidebar.number_input('BMI- Obese (Yes:1, No:0)',min_value=0, max_value=1, value=0)
        BMI_Category_Overweight  = st.sidebar.number_input('BMI-Overweight  (Yes:1, No:0)',min_value=0, max_value=1, value=0)

        data = {
            'Age':float(Age),
            'Sleep Duration': float(Sleep_Duration),
            'Quality of Sleep':float(Quality_of_Sleep),
            'Physical Activity Level':float(Physical_Activity_Level),
            'Stress Level': float(Stress_Level),
            'Heart Rate': float(Heart_Rate),
            'Daily Steps': float(Daily_Steps),
            'Systolic': float(Systolic),
            'Diastolic': float(Diastolic),
            'Gender_Female': int(Gender_Female),
            'Gender_Male': int(Gender_Male),
            'Occupation_Accountant': int(Occupation_Accountant),
            'Occupation_Doctor': int(Occupation_Doctor),
            'Occupation_Engineer': int(Occupation_Engineer),
            'Occupation_Lawyer': int(Occupation_Lawyer),
            'Occupation_Manager': int(Occupation_Manager),
            'Occupation_Nurse': int(Occupation_Nurse),
            'Occupation_Sales Representative': int(Occupation_Sales_Representative),
            'Occupation_Salesperson': int(Occupation_Salesperson),
            'Occupation_Scientist': int(Occupation_Scientist),
            'Occupation_Software Engineer': int(Occupation_Software_Engineer),
            'Occupation_Teacher': int(Occupation_Teacher),
            'BMI Category_Normal': int(BMI_Category_Normal),
            'BMI Category_Obese': int(BMI_Category_Obese),
            'BMI Category_Overweight':int(BMI_Category_Overweight)
        }

        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_features()

    st.subheader('User Input Parameters')
    st.write(df)


    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.write("""
    ### Class labels and their corresponding index number
             
    Insommia  -  0
             
    No disorder - 1
             
    Sleep Apnea - 2   
    """)

    st.subheader('Prediction')
    st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

if __name__ == '__main__':
    main()

