import streamlit as st
import pandas as pd
import numpy as np, os,json
import tensorflow as tf
import models.translation_prediction as tp
from collections import Counter


translator_model = tf.keras.models.load_model('./artifacts/translator_model.h5', compile=True)

current_dir = os.getcwd()
txt_folder = os.path.join(current_dir,"data","txt")
dict_folder = os.path.join(current_dir,"data","dict")
    


with open(os.path.join(dict_folder,"target_dict.txt"),'r',encoding='utf-8') as f:
    target_dict = json.load(f)


target_dict = {v: k for k, v in target_dict.items()}


st.title('English-2-Hindi Translator')
text_box = st.text_input("Enter English Scentence:")

if st.button("Submit"):
    translator_pred = tp.translation_prediction([text_box])
    sequences = translator_pred.load_sequences()
    output_array=translator_model.predict(sequences)
    print("output array",output_array)
    indices_hindi = np.argmax(output_array,axis=1)[0]
    print("indices hindi",indices_hindi)
    
    print(Counter(indices_hindi).most_common())
    padding_token = Counter(indices_hindi).most_common(1)[0][0]

    print(padding_token)

    high_freq_tokens = [index for index, freq in Counter(indices_hindi).items() if freq > 4]
    translated_words = [target_dict.get(index, '') for index in indices_hindi if index not in high_freq_tokens and index in target_dict]
    print(f"indices : {translated_words}")
    
    translated_words = ' '.join(translated_words)

    st.write(translated_words)

