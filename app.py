import streamlit as st
from transformers import AutoModelForSeq2SeqLM

# Load the pre-trained Transformer model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Define a function to translate a French sentence into English
def translate(sentence):

  # Convert the French sentence to a tensor
  sentence_tensor = torch.tensor(sentence)
  
  translated_sentence = model.generate(input_ids=sentence, max_length=128)
  return translated_sentence[0]

# Create a Streamlit app
st.title("French to English Translator")

# Get the French sentence from the user
french_sentence = st.text_input("Enter a French sentence:")

# Translate the French sentence into English
english_sentence = translate(french_sentence)

# Display the translated sentence to the user
st.write("English translation:", english_sentence)
