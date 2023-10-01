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
    height=None,
    max_chars=None,
    key="text_area",
)

if language == "French to English":
  fr_en_translation_model_name = 'Helsinki-NLP/opus-mt-fr-en'
  fr_en_model = MarianMTModel.from_pretrained(fr_en_translation_model_name)
  fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_translation_model_name)

  fr_en_batch = fr_en_tokenizer.prepare_seq2seq_batch(src_texts=[text])

  # Extract the input_ids field from the fr_en_batch object
  fr_en_input_ids = fr_en_batch['input_ids']

  # Convert the fr_en_input_ids variable to a tensor
  fr_en_input_ids = torch.as_tensor(fr_en_input_ids, dtype=torch.long, device='cpu')

  if st.button('Translate to English', key="translate_to_english"):
    fr_en_gen = fr_en_model.generate(input_ids=fr_en_input_ids)
    fr_en_translation = fr_en_tokenizer.batch_decode(fr_en_gen, skip_special_tokens=True)
    st.markdown(f"**English translation:** {fr_en_translation[0]}")

else:
  en_fr_translation_model_name = 'Helsinki-NLP/opus-mt-en-fr'
  en_fr_model = MarianMTModel.from_pretrained(en_fr_translation_model_name)
  en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_translation_model_name)

  en_fr_batch = en_fr_tokenizer.prepare_seq2seq_batch(src_texts=[text])

  # Extract the input_ids field from the en_fr_batch object
  en_fr_input_ids = en_fr_batch['input_ids']

  # Convert the en_fr_input_ids variable to a tensor
  en_fr_input_ids = torch.as_tensor(en_fr_input_ids, dtype=torch.long, device='cpu')

  if st.button('Translate to French', key="translate_to_french"):
    en_fr_gen = en_fr_model.generate(input_ids=en_fr_input_ids)
    en_fr_translation = en_fr_tokenizer.batch_decode(en_fr_gen, skip_special_tokens=True)
    st.markdown(f"**French translation:** {en_fr_translation[0]}")
