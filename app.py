
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from joblib import load
from sklearn import preprocessing

model = joblib.load('model.pkl')

def predict(model, input_df):
    def label_encoder(df):
        cat_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object'] ]
        labelencoder = preprocessing.LabelEncoder()
        for cat in cat_cols:    
            df[cat] = labelencoder.fit_transform(df[cat])            
        return df
    final_df = label_encoder(input_df)
    predictions_df = model.predict(final_df)
    prediction = predictions_df[0]
    prediction_rounded = round(prediction,2)
    return prediction_rounded

def run():

    from PIL import Image
    #image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('''This app is created to predict patient hospital charges
                    in order to help medical insurance make decisions on charging premiums.
                    Training dataset obtained from Kaggle.''')
    #st.sidebar.success('https://www.pycaret.org')
    st.sidebar.success('analyst@harveybsmith.com')
    
    st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Online':

        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = model.predict(data)
            st.write(predictions)

if __name__ == '__main__':
    run()
