import tensorflow as tf
import os
import pathlib
import numpy as np
import zipfile 
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_util
import streamlit as st
from datetime import datetime
from streamlit.uploaded_file_manager import UploadedFile

from run_object_detection import run_object_detection

def save_uploaded_file(directory, file) :
        # 1.디렉토리가 있는지 확인하여, 없으면 디렉토리부터만든다.
            if not os.path.exists(directory) :
                os.makedirs(directory)
            # 2. 디렉토리가 있으니, 파일을 저장.
            with open(os.path.join(directory, file.name), 'wb') as f :
                f.write(file.getbuffer())
            return st.success("Saved file : {} in {}".format(file.name, directory))


def main():
    st.title('Tensorflow Object Detection')

    menu = ['Object Detection', 'Text Detection']

    choice = st.sidebar.selectbox('메뉴 선택', menu)

    if choice == 'Object Detection' :
        run_object_detection()
if __name__ == '__main__':
    main()