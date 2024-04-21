# MIT License
#
# Copyright (c) 2024 Tsutomu Furuse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import streamlit as st
from PIL import Image
import numpy as np
import requests
import io
from transformers import AutoProcessor, AutoModel
import torch
import pandas as pd
import random


HF_MODEL = "google/siglip-base-patch16-256-multilingual"
SRC_SEL_NET = "Network :globe_with_meridians:"
SRC_SEL_FILE = "Local File :open_file_folder:"
SRC_SEL_CAM = "Camera :camera:"


class ClipInferenceModel():

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() \
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(HF_MODEL)
        self.processor = AutoProcessor.from_pretrained(HF_MODEL)

    def infer(self, image, texts):
        image = image.convert("RGB")
        inputs = self.processor(
            text=texts, images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        return logits_per_image, probs
    

@st.cache_resource
def get_model():
    model = ClipInferenceModel()
    return model


def infer(image):
    if image is None:
        return
    
    key=str(st.session_state["text_input_key"])
    target_text = st.session_state[key]
    
    df = st.session_state["template"]

    texts = df["Target Text"].tolist()
    texts.append(target_text)
    
    print("Inference")
    logits, probs = model.infer(image, texts)
    logits_data = logits.data[0].tolist()
    num_lines = len(logits_data)
    if num_lines <= 0:
        return

    df = pd.DataFrame({
        "Target Text": texts,
        "Logit": logits_data,
        "Probability": probs.data[0].tolist()
    })

    st.session_state["template"] = df
    st.session_state["data_editor_key"] = random.randint(0, 100000)
    st.session_state["text_input_key"] = random.randint(0, 100000)


image = None

model = get_model()
                  
if "template" not in st.session_state:
    df = pd.DataFrame(columns=["Target Text", "Logit", "Probability"])
    st.session_state["template"] = df

if "data_editor_key" not in st.session_state:
    st.session_state["data_editor_key"] = random.randint(0, 100000)

if "text_input_key" not in st.session_state:
    st.session_state["text_input_key"] = random.randint(0, 100000)

with st.sidebar:
    sel = st.radio(
        label="Select Image Source",
        options=[
            SRC_SEL_NET, 
            SRC_SEL_FILE, 
            SRC_SEL_CAM
        ],
        captions=[
            "Download an image data from the specified URL",
            "Select an image file from the local drive",
            "Capture an image from the computer camera"
        ]
    )

if sel == SRC_SEL_NET:
    url = st.text_input('The URL Link')
    if url:
        image = io.BytesIO(requests.get(url).content)
        if image:
            st.image(image)
elif sel == SRC_SEL_FILE:
    image = st.file_uploader("Choose an Image File", type=["png", "jpg"])
    if image:
        st.image(image)
elif sel == SRC_SEL_CAM:
    image = st.camera_input("Take a Picture")

if image:
    image = Image.open(image).convert("RGB")

st.text_input(
    "Target Text", 
    key=str(st.session_state["text_input_key"]),
    on_change=infer,
    args=(image,)
)
# st.button("Submit", on_click=infer, args=(image,))

df_editor = st.data_editor(
    st.session_state["template"], 
    num_rows="dynamic",
    disabled=True,
    key=str(st.session_state["data_editor_key"]),
)
