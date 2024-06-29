import streamlit as st
import cv2
import os
from openai import OpenAI
import numpy as np
import pandas as pd
import cv2
import base64
import json
import easyocr
reader = easyocr.Reader(['ja'], gpu=False)

# import pytesseract

# from paddleocr import PaddleOCR

# ocr = PaddleOCR(
#         use_gpu=False,
#         lang = "japan",
#     )

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", st.secrets["OpenAI"]["API_KEY"]))

st.set_page_config(page_title="Camera OCR", page_icon="📷")
st.title("📷 Camera OCR")
st.write(
    """
    This app get a text from camera picture.
    """
)

img_file_buffer = st.camera_input("Take a picture")

def img_to_base64(img, resize=300):
    h,w,_ = img.shape
    if h>w and h>resize:
        img = cv2.resize(img, (int(resize*w/h), resize))
    elif w>h and w>resize:
        img = cv2.resize(img, (resize, int(resize*h/w)))

    _, encoded = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(encoded).decode("utf-8")
    
    st.write("jpg: {0}".format(len(img_str)))

    return img_str



def image_to_ocred_text(img):
    ocr_result = reader.readtext(img)
    ocred_text_list = []
    for res in ocr_result:
        ((x1, y1), _, _, _), temp_text, _ = res
        temp_ocred_text = f"{x1:.0f} {y1:.0f} {temp_text}"
        ocred_text_list.append(temp_ocred_text)
    return "/n".join(ocred_text_list)


if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    img_cv2 = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_base64 = img_to_base64(img_cv2)

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(img_cv2.shape)
    # st.write(img_base64)

    ocred_text = image_to_ocred_text(img_cv2)
    st.write("hello")
    st.write(ocred_text[:100])
    

    schema = {
        "建物名": "string",
        "住所": "string",
        "構造": "string",
        "建築年":"integer",
        "トイレの有無":"boolean",
        "冷暖房設備の有無":"boolean",
        "駐車場の有無":"boolean",
    }

    prompt = f"""
    あなたは不動産の専門家で、賃貸借契約書の情報を綺麗に整理してもらいます。
    以下に、OCRしたテキストとBase64 foramtの画像があります。ここからoutput schemaの情報を抽出してjsonで出力してください。
    設備の有無の情報はBase64 stringを優先して、テキスト情報はOCRテキストを使ってください。

    # Base64 image string
    {img_base64}
    
    # OCR text
    {ocred_text}

    # output schema
    {schema}
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

    # st.write(json.loads(response.choices[0].message.content))
    st.write(response)