import streamlit as st
from txtai.pipeline import Summary
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from langdetect import detect
import nlpaug.augmenter.word as naw
from collections import Counter
import re
from PIL import Image
import pytesseract
import pandas as pd
from docx import Document
import io

st.set_page_config(layout="wide")

@st.cache_resource
def text_summary(text, maxlength=None):
    # create summary instance
    summary = Summary()
    text = (text)
    result = summary(text)
    return result

@st.cache_resource
def translate_text(text, target_lang="en"):
    detected_lang = detect(text)
    translation = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
    return translation

@st.cache_resource
def paraphrase_text(text):
    # create paraphraser instance
    aug = naw.SynonymAug()
    augmented_text = aug.augment(text)
    # Join list elements into a single string if the result is a list
    if isinstance(augmented_text, list):
        augmented_text = " ".join(augmented_text)
    return augmented_text

def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")


def extract_text_from_docx(uploaded_file):
    document = Document(uploaded_file)
    text = "\n".join([para.text for para in document.paragraphs])
    return text

@st.cache_resource
def extract_text(uploaded_file, file_type):
    if file_type == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(uploaded_file)
    elif file_type == "txt":
        return extract_text_from_txt(uploaded_file)
    elif file_type == "docx":
        return extract_text_from_docx(uploaded_file)
    else:
        return ""

def is_dialogue(text):
    # Check for presence of quotation marks or speaker labels
    dialogue_markers = ['"', "'", ":", "-"]
    return any(marker in text for marker in dialogue_markers)

def detect_primary_speaker(text):
    # Find all capitalized words not at the start of a sentence
    words = re.findall(r'\b[A-Z][a-z]*\b', text)
    # Count the frequency of each word
    word_counts = Counter(words)
    if word_counts:
        # Get the most common word
        primary_speaker = word_counts.most_common(1)[0][0]
    else:
        primary_speaker = "Unknown Speaker"
    return primary_speaker


choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Translate Text", "Paraphrase Text"])

if choice == "Summarize Text":
    st.subheader("Summarize Text using txtai")
    upload_option = st.radio("Choose input method", ("Text Area", "Upload File"))
    if upload_option == "Text Area":
        input_text = st.text_area("Enter your text here")
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf", "png", "jpg", "jpeg", "txt", "docx"])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            input_text = extract_text(uploaded_file, file_type)


    if 'input_text' in locals() and input_text:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = text_summary(input_text)
                if is_dialogue(input_text):
                    primary_speaker = detect_primary_speaker(input_text)
                    result += f"\n\nPrimary Speaker: {primary_speaker}"
                st.success(result)
                st.download_button(label="Download Result as Text File", data=result, file_name="summary.txt", mime="text/plain")



elif choice == "Translate Text":
    st.subheader("Translate Text using Google Translate")
    input_text = st.text_area("Enter your text here")
    target_lang_map = {"English": "en","Telugu": "te", "Hindi": "hi", "Urdu": "ur", "Arabic": "ar", "French": "fr", "Spanish": "es", "Tamil": "ta", "Chinese": "zh-CN"}
    if input_text:
        detected_lang = detect(input_text)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.selectbox("Detected language", options=list(target_lang_map.keys()), index=list(target_lang_map.values()).index(detected_lang))
        with col2:
            target_lang = st.selectbox("Select target language", ["English","Telugu", "Hindi", "Urdu", "Arabic", "French", "Spanish", "Tamil", "Chinese"])
        if st.button("Translate Text"):
            col3, col4 = st.columns([1, 1])
            with col3:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col4:
                st.markdown("**Translation Result**")
                translated_text = translate_text(input_text, target_lang=target_lang_map[target_lang])
                st.success(translated_text)
                st.download_button(label="Download Result as Text File", data=translated_text, file_name=f"translate_{target_lang}.txt", mime="text/plain")



elif choice == "Paraphrase Text":
    st.subheader("Paraphrase Text using NLPAug")
    upload_option = st.radio("Choose input method", ("Text Area", "Upload File"))
    if upload_option == "Text Area":
        input_text = st.text_area("Enter your text here")
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type=[["pdf", "png", "jpg", "jpeg", "txt", "csv", "docx"]])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            input_text = extract_text(uploaded_file, file_type)


    if 'input_text' in locals() and input_text:
        if st.button("Paraphrase Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Paraphrase Result**")
                result = paraphrase_text(input_text)
                st.success(result)
                st.download_button(label="Download Result as Text File", data=result, file_name="paraphrase.txt", mime="text/plain")








