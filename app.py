import torch
from transformers import MarianTokenizer, MarianMTModel
import streamlit as st

st.title('French <--> English Translator')

with st.sidebar:
  language = st.radio(
    "Translation direction:",
    ["French to English", "English to French"],
    index=0,
  )

text = st.text_area(
    "Enter Text:",
    default="Bonjour, le monde!",
    height=None,
    max_chars=None,
    key="text_area",
)

if language == "French to English":
  fr_en_translation_model_name = 'Helsinki-NLP/opus-mt-fr-en'
  fr_en_model = MarianMTModel.from_pretrained(fr_en_translation_model_name)
  fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_translation_model_name)

  if st.button('Translate to English', key="translate_to_english"):
    fr_en_batch = fr_en_tokenizer.prepare_seq2seq_batch(src_texts=[text])
    fr_en_gen = fr_en_model.generate(**fr_en_batch)
    fr_en_translation = fr_en_tokenizer.batch_decode(fr_en_gen, skip_special_tokens=True)
    st.markdown(f"**English translation:** {fr_en_translation[0]}")

else:
  en_fr_translation_model_name = 'Helsinki-NLP/opus-mt-en-fr'
  en_fr_model = MarianMTModel.from_pretrained(en_fr_translation_model_name)
  en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_translation_model_name)

  if st.button('Translate to French', key="translate_to_french"):
    en_fr_batch = en_fr_tokenizer.prepare_seq2seq_batch(src_texts=[text])
    en_fr_gen = en_fr_model.generate(**en_fr_batch)
    en_fr_translation = en_fr_tokenizer.batch_decode(en_fr_gen, skip_special_tokens=True)
    st.markdown(f"**French translation:** {en_fr_translation[0]}")
