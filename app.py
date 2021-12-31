import tensorflow as tf
import os
import pathlib
import numpy as np
import zipfile 
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import streamlit as st
from datetime import datetime
from streamlit.uploaded_file_manager import UploadedFile

from tf_object_detection import run_object_detection
from text_detection import run_text_detection

def main():

    menu = ['Object Detection', 'Text Detection']

    choice = st.sidebar.selectbox('메뉴 선택', menu)

    if choice == 'Object Detection' :
        run_object_detection()
    if choice == 'Text Detection' :
        run_text_detection()
if __name__ == '__main__':
    main()