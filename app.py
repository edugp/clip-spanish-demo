import os
import sys

import streamlit as st
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

LOCAL_PATH = snapshot_download("flax-community/clip-spanish")
sys.path.append(LOCAL_PATH)

from modeling_hybrid_clip import FlaxHybridCLIP
from test_on_image import run_inference


def save_file_to_disk(uplaoded_file):
    temp_file = os.path.join("/tmp", uplaoded_file.name)
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file


@st.cache(
    hash_funcs={
        transformers.models.bert.tokenization_bert_fast.BertTokenizerFast: id,
        FlaxHybridCLIP: id,
    }
)
def load_tokenizer_and_model():
    # load the saved model
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    model = FlaxHybridCLIP.from_pretrained(LOCAL_PATH)
    return tokenizer, model


tokenizer, model = load_tokenizer_and_model()

st.title("Caption Scoring")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])
text_input = st.text_input("Type a caption")

if uploaded_file is not None and text_input:
    local_image_path = None
    try:
        local_image_path = save_file_to_disk(uploaded_file)
        score = run_inference(local_image_path, text_input, model, tokenizer).tolist()
        st.image(
            uploaded_file,
            caption=text_input,
            width=None,
            use_column_width=None,
            clamp=False,
            channels="RGB",
            output_format="auto",
        )
        st.write(f"## Score: {score:.2f}")
    finally:
        if local_image_path:
            os.remove(local_image_path)
