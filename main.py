from operator import index
import streamlit as st
import plotly.express as px
from pydantic_settings import BaseSettings
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import torch 
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import tempfile
import numpy as np
from facenet_pytorch import MTCNN
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
@st.cache_resource
def torch_model():
    return torch.hub.load('ultralytics/yolov5','yolov5s')
model = torch_model()
@st.cache_resource
def emotion_model():
    model_name='enet_b0_8_best_afew'
    return HSEmotionRecognizer(model_name=model_name,device='cpu')
fer= emotion_model()
@st.cache_resource
def face_model():
    return MTCNN(keep_all=False, post_process=False, min_face_size=40, device='cpu')
mtcnn = face_model()
def detect_face(frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    bounding_boxes=bounding_boxes[probs>0.9]
    return bounding_boxes
def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # Categorical or text columns
            df[col].fillna('', inplace=True)  # Fill with empty string for text or specific value for categories
        else:  # Numeric columns
            df[col].fillna(df[col].mean(), inplace=True)  # Fill with mean for numeric columns
    return df

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
st.markdown(
    """

    <style>
    .stApp {
        background: #c5c5c5;
    }
    .stApp stImage {
        max-width: 50%;
    }
    .stApp stRadio {
        background-color: #c5c5c5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Web App by Bro Code")
    choice = st.radio("Navigation", ["Upload","Profiling","Data Manipulation","Modelling", "Download","Deep Learning"])

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
@st.cache_data
def profiling_df(datafr):
    profile_df = datafr.profile_report()
    return profile_df

if choice == "Profiling": 
    st.title("Data Analysis(This will take time Based on the size of the Dataframe Provided)")
    st_profile_report(profiling_df(df))
if choice =="Data Manipulation":
    st.title("Data Manipulation")
    lists = ['Missing Values','Encode Categorical Variables','Feature Scaling','Handling Imbalanced Data']
    chosen_target = st.selectbox('Choose the Target Column(Target Column is the one that is Variable for which the predictions are made)', lists)
    st.markdown("PLEASE SELECT WHAT MANIPULATION I CAN DO TO THE DATA FOR U!")
    if chosen_target == 'Missing Values':
        st.header('BEFORE')
        st.dataframe(df)
        st.button('Manipulate!')
        # Drop rows with missing values
        df_filled = handle_missing_values(df)
        st.header('AFTER')
        st.dataframe(df_filled)
    elif chosen_target =='Encode Categorical Variables':
        st.header('BEFORE')
        st.dataframe(df)
        st.button('Manipulate!')
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        st.header('AFTER')
        st.dataframe(df_encoded)
    elif chosen_target == 'Feature Scaling':
        scale_col = st.selectbox('Choose which columns missing values you want to deal with)', df.columns)
        st.header('BEFORE')
        st.dataframe(df)
        st.button('Manipulate!')
        try:              
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            st.header("AFTER")
            st.dataframe(df_scaled)
        except ValueError:
            st.header("Select a suitable feature")

    else:
        target = st.selectbox('Choose the Y(Prediction/target)Variable', df.columns)
        st.header('BEFORE')
        st.dataframe(df)
        st.button('Manipulate!')
        X = df.drop(target, axis=1)
        y = df[target]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        resampled_df = pd.DataFrame(data=X_resampled, columns=X.columns)
        resampled_df['target'] = y_resampled
        st.header('AFTER')
        st.data(resampled_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column(Target Column is the one that is Variable for which the predictions are made)', df.columns)
    if st.button('Run Modelling'): 
        
        setup(data=df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        st.header('These are some Models that I have experimented On Hope these are enough :smile:')
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
if choice == "Deep Learning":
    st.title("Deep Learning Use Cases")
    tasks = ["Object Detection on Image ","Emotion Detection(Image)"]
    choosen_task = st.selectbox('Choose the Target Column(Target Column is the one that is Variable for which the predictions are made)', tasks)   
    if choosen_task == tasks[0]:
        file = st.file_uploader("Upload Your Image(png/jpg/jpeg)",type=['png', 'jpg', 'jpeg'])
        if file is not None:
            st.header("This is your Uploaded file")
            dis = file.read()
            st.image(dis)
            if st.button('Run Model'):
                image = Image.open(file)
                results = model(image)
                st.write("Detected Objects:")
                objects = results.pandas().xyxy[0]
                st.dataframe(objects)  
                drawn_image = results.render()[0]
                st.image(drawn_image)     
    else:
        file = st.file_uploader("Upload Your Image(png/jpg/jpeg)",type=['png', 'jpg', 'jpeg'])
        if file is not None:
            st.header("This is your Uploaded file")
            dis = file.read()
            st.image(dis)
            if st.button('U want me to Find the emotion '):
                pil_image = Image.open(file)
                np_image = np.array(pil_image)
                frame = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                bounding_boxes=detect_face(frame)
                for bbox in bounding_boxes:
                    box = bbox.astype(int)
                    x1,y1,x2,y2=box[0:4]    
                    face_img=frame[y1:y2,x1:x2,:]
                    emotion,scores=fer.predict_emotions(face_img,logits=True)
                    st.image(dis)
                    st.header('')










