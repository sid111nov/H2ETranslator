from .data_tokenization import  data_tokenization
import os, json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from .data_tokenization import data_tokenization as DataTokenization 

class translation_prediction:
    
    
    current_dir = os.getcwd()
    txt_folder = os.path.join(current_dir,r"data\\txt")
 
    

    def __init__(self,scentences):
        self.scentences = scentences
        source_dict = {}
        current_dir = os.getcwd()
        self.tokenizer_instance = DataTokenization()
        
        self.source_dict = self.tokenizer_instance.get_source_dict()
        self.maxlen = self.tokenizer_instance.max_length()

    def load_sequences(self):
        token_list = []
        
        for line in self.scentences:
            token_list.append(self.tokenizer_instance.tokenizer(line,self.source_dict))
        token_list = self.tokenizer_instance.get_padded_array(token_list,self.maxlen)   
        return token_list
    
