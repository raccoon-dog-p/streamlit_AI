import io
import os
import tensorflow as tf
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import streamlit as st
from datetime import datetime
from streamlit.uploaded_file_manager import UploadedFile
import requests

from Translation import get_translate

def save_uploaded_file(directory, file) :
    # 1.디렉토리가 있는지 확인하여, 없으면 디렉토리부터만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니, 파일을 저장.
    with open(os.path.join(directory, file.name), 'wb') as f :
        f.write(file.getbuffer())
    return st.success("Saved file : {} in {}".format(file.name, directory))


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\pys1\\Downloads\\emerald-rhythm-332904-7dd050962f9f.json'
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    text_list = list(map(lambda x: x.description, texts))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return text_list

def run_text_detection():
    st.title('TEXT DETECTION BY GOOGLE CV API')
    uploaded_files = st.file_uploader('이미지파일 업로드',type=['png','jpeg','jpg']
                        ,accept_multiple_files=True)
    
    if uploaded_files is not None:

        for file in uploaded_files:
            save_uploaded_file('temp_files',file)

            img = Image.open(file)
            st.image(img)
            # detect_text(file)
            PATH_TO_IMAGE_DIR = pathlib.Path('temp_files')
            IMAGE_PATHS_JPG = list(PATH_TO_IMAGE_DIR.glob('*.jpg'))
            IMAGE_PATHS_PNG= list(PATH_TO_IMAGE_DIR.glob('*.png'))

        if st.button('실행'):
            print(IMAGE_PATHS_JPG[0])
            if uploaded_files[0].name in str(IMAGE_PATHS_JPG[0]):
                st.subheader('추출된 텍스트')
                text=(detect_text(IMAGE_PATHS_JPG[0]))
                st.write(text[0])
                st.subheader('영어 번역')
                translate = get_translate(text[0])
                st.write(translate)
            elif uploaded_files[0].name in str(IMAGE_PATHS_PNG[0]):
                st.subheader('추출된 텍스트')
                text=(detect_text(IMAGE_PATHS_PNG[0]))
                st.write(text[0])
                st.subheader('영어 번역')
                translate = get_translate(text[0])
                st.write(translate)
    