import streamlit as st
import cv2
import os
from openai import OpenAI
import numpy as np
import pandas as pd
import cv2
import base64
import json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", st.secrets["OpenAI"]["API_KEY"]))

st.set_page_config(page_title="Camera OCR", page_icon="📷")
st.title("📷 Camera OCR")
st.write(
    """
    This app get a text from camera picture.
    """
)

img_file_buffer = st.camera_input("Take a picture")

def img_to_base64(img, resize=400):
    h,w,_ = img.shape
    if h>w:
        img = cv2.resize(img, (int(resize*w/h), resize))
    else:
        img = cv2.resize(img, (resize, int(resize*h/w)))
    _, encoded = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(encoded).decode("utf-8")

    return img_str


if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    img_cv2 = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_base64 = img_to_base64(img_cv2)

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(img_cv2.shape)
    st.write(img_base64)

    prompt = f"""
    以下のBase64 形式の画像を読み込んでなにが書いてあるか説明してください。

    # Base64 string
    {img_base64[:10000]}

    # 出力
    """    

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
        }
    ],
    max_tokens=300,
    )

    st.write(response.choices[0].message.content)