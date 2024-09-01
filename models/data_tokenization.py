import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np


class data_tokenization:

    def __init__(self):
            
        current_dir = os.getcwd()
        txt_folder = os.path.join(current_dir,r"data\\txt")
        dict_folder = os.path.join(current_dir,r"data\\dict")
        padded_folder = os.path.join(current_dir,r"data\\padded")



        self.current_dir = current_dir
        self.txt_folder = txt_folder
        self.dict_folder = dict_folder
        self.padded_folder = padded_folder
        

        

             

        #loading dictionaries
    def get_source_dict(self):
        source_dict = {}
        with open(os.path.join(self.dict_folder,"english_dictonary.txt"),'r',encoding='utf-8') as f:
            source_dict = json.load(f)
        return source_dict    

    #get hindi dictionary
    def get_target_dict(self):
        target_dict = {}
        with open(os.path.join(self.dict_folder,"hindi_dictonary.txt"),'r',encoding='utf-8') as f:
            target_dict = json.load(f)
        return target_dict
   
    #get maximum possible length of scentence from a scentence list
    def max_length(self):
        max_scent_len = 0
        for filename in ['source_full.txt','target_full.txt']:
            with open(os.path.join(self.txt_folder,filename),'r',encoding='utf-8') as f:
                for line in f:
                    if(len(line)>max_scent_len):
                        max_scent_len = len(line)
        return  max_scent_len

    #getting pre-padded array, can be post as well
    def get_padded_array(self,seq,maxlength):
        sequence_padded = pad_sequences(seq, maxlen=maxlength, padding='pre') 
        return np.array(sequence_padded)

    #loading scentence from file
    def load_token_sequences(self,filename,dict):
        with open(os.path.join(self.txt_folder,filename),'r',encoding='utf-8') as f:
            seq = []
            for line in f:
                seq.append(self.tokenizer(line,dict))
            return seq    
        
    #tokenizing scentence
    def tokenizer(self,line,dict):
        seq=[]
        for word in line.split():
            seq.append(dict.get(word,1))
        return seq   

       