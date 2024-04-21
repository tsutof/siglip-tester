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
import pandas as pd
import random
from siglip_collator import *


SRC_SEL_NET = "Network :globe_with_meridians:"
SRC_SEL_FILE = "Local File :open_file_folder:"
SRC_SEL_CAM = "Camera :camera:"
SRC_SEL_NONE = "No image source selection"
SESS_KEY_DF = "dataframe"
SESS_KEY_INPUT_KEY = "target_text_key"
SESS_KEY_LAST_SOURCE = "last_source"


@st.cache_resource
def get_model():
    model = SiglipModel()
    return model


def initialize_session_data():
    # Initialize the dataframe to hold the results                  
    if SESS_KEY_DF not in st.session_state:
        clear_results()

    # Initialize the texe input key
    if SESS_KEY_INPUT_KEY not in st.session_state:
        st.session_state[SESS_KEY_INPUT_KEY] = random.randint(0, 100000)

    # Initialize the session state for the last input source selection
    if SESS_KEY_LAST_SOURCE not in st.session_state:
        st.session_state[SESS_KEY_LAST_SOURCE] = SRC_SEL_NONE


def clear_results():
    df = pd.DataFrame(columns=["Target Text", "Logit", "Probability"])
    st.session_state[SESS_KEY_DF] = df


# Callback when the new target text input
def update(model, image):
    if image is None:
        return
    
    # Get the last input target text
    key=str(st.session_state[SESS_KEY_INPUT_KEY])
    target_text = st.session_state[key]
    text_list = [target_text]
    
    # Do inference by SigLIP
    logits, probs = model.infer(image, text_list)

    # Get the number of results the dataframe currently holds
    df = st.session_state[SESS_KEY_DF]
    n = len(df)

    # Costruct a DataFrame with the inference result
    logit_list = logits.data[0].tolist()
    prob_list = probs.data[0].tolist()
    df2 = pd.DataFrame(
        {
            "Target Text": text_list,
            "Logit": logit_list,
            "Probability": prob_list
        },
        index=[n + 1]
    )
    
    # concatenate the existing results with the last result
    df = df2 if df.empty else pd.concat([df, df2])
    st.session_state[SESS_KEY_DF] = df

    # Regenerate the text input key to clear the last input
    st.session_state[SESS_KEY_INPUT_KEY] = random.randint(0, 100000)


image = None

# Load the SigLIP model
model = get_model()

# Initialize the session data
initialize_session_data()

# Sidebar for the image source selection
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

st.title("SigLIP Collator Application")

if sel != st.session_state[SESS_KEY_LAST_SOURCE]:
    clear_results()

if sel == SRC_SEL_NET:
    # Download an image from internet
    url = st.text_input(
        "Image URL Link", 
        on_change=clear_results
    )
    if url:
        image = io.BytesIO(requests.get(url).content)
        if image:
            st.image(image)
elif sel == SRC_SEL_FILE:
    # Load an image from the local drive
    image = st.file_uploader(
        "Choose an Image File", 
        type=["png", "jpg"],
        on_change=clear_results
    )
    if image:
        st.image(image)
elif sel == SRC_SEL_CAM:
    # Capture an image from the computer camera
    image = st.camera_input(
        "Take a Picture",
        on_change=clear_results
    )

if image:
    # Convert the image into the PIL format
    image = Image.open(image)

    # Text input widget to specify a target text
    st.text_input(
        "Target Text", 
        key=str(st.session_state[SESS_KEY_INPUT_KEY]),
        on_change=update,
        args=(model, image)
    )

    # Button widget to clear the results
    st.button("Clear", on_click=clear_results)

    # Data frame widget to hold the results
    df_editor = st.dataframe(
        st.session_state[SESS_KEY_DF],
        use_container_width=True
    )
else:
    # No image loaded
    st.write("Load an image first")

st.session_state[SESS_KEY_LAST_SOURCE] = sel
