import torch
from transformers import MarianTokenizer, MarianMTModel
import streamlit as st

st.title('French <--> English Translator')
text = st.text_area("Enter Text:", value='', height=None, max_chars=None, key=None)

if st.button('Translate to English'):
    if text == '':
        st.write('Please enter French text for translation') 
    else: 
        fr_en_translation_model_name = 'Helsinki-NLP/opus-mt-fr-en'
        fr_en_model = MarianMTModel.from_pretrained(fr_en_translation_model_name)
        fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_translation_model_name)
        fr_en_batch = fr_en_tokenizer.prepare_seq2seq_batch(src_texts=[text])
        fr_en_gen = fr_en_model.generate(**fr_en_batch)
        fr_en_translation = fr_en_tokenizer.batch_decode(fr_en_gen, skip_special_tokens=True)
        st.write('', str(fr_en_translation).strip('][\''))
else: pass

if st.button('Translate to French'):
    if text == '':
        st.write('Please enter English text for translation') 
    else: 
        en_fr_translation_model_name = 'Helsinki-NLP/opus-mt-en-fr'
        en_fr_model = MarianMTModel.from_pretrained(en_fr_translation_model_name)
        en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_translation_model_name)
        en_fr_batch = en_fr_tokenizer.prepare_seq2seq_batch(src_texts=[text])
        en_fr_gen = en_fr_model.generate(**en_fr_batch)
        en_fr_translation = en_fr_tokenizer.batch_decode(en_fr_gen, skip_special_tokens=True)
        st.write('', str(en_fr_translation).strip('][\''))
else: pass
