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
        with open(os.path.join(self.dict_folder,"source_dict.txt"),'r',encoding='utf-8') as f:
            source_dict = json.load(f)
        return source_dict    

    #get hindi dictionary
    def get_target_dict(self):
        target_dict = {}
        with open(os.path.join(self.dict_folder,"target_dict.txt"),'r',encoding='utf-8') as f:
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

        #getting tokenized sequence for scentences and 

        # def write_padded_seq_to_file(source_file,dict, padded_file):
        #     with open(os.path.join(txt_folder,source_file),'r',encoding='utf-8') as f:
        #         with open(os.path.join(padded_folder,padded_file),'w',encoding='utf-8') as t:

        #             for line in f:
        #                 seq = tokenizer(line,dict)
                        
        #                 #padded_seq = pad_sequences([seq], maxlen=max_scent_len, padding='post')
        #                 t.write(' '.join(map(str, [seq])) + '\n')
            
        # write_padded_seq_to_file("source_train.txt",source_dict,"source_padded_train.txt")
        # write_padded_seq_to_file("target_train.txt",target_dict,"target_padded_train.txt")

        # write_padded_seq_to_file("source_test.txt",source_dict,"source_padded_test.txt")
        # write_padded_seq_to_file("target_test.txt",target_dict,"target_padded_test.txt")

        # write_padded_seq_to_file("source_valid.txt",source_dict,"source_padded_valid.txt")
        # write_padded_seq_to_file("target_valid.txt",target_dict,"target_padded_valid.txt")

        # source_seq_tr = load_token_sequences(os.path.join(txt_folder,"source_train.txt"),source_dict)

        # target_seq_tr = load_token_sequences(os.path.join(txt_folder,"target_train.txt"),target_dict)

        #padding sequences
                
        # source_sequence_padded_tr = pad_sequences(source_seq_tr, maxlen=max_scent_len, padding='post') 
        # target_sequence_padded_tr = pad_sequences(target_seq_tr, maxlen=max_scent_len, padding='post')         

        # Converting to tensors
        # source_sequences_tensor_tr = tf.convert_to_tensor(source_sequence_padded_tr)
        # target_sequences_tensor_tr = tf.convert_to_tensor(target_sequence_padded_tr)


        #Scentece to tensors validation Data 
        # source_seq_v = load_token_sequences(os.path.join(txt_folder,"source_valid.txt"),source_dict)

        # target_seq_v = load_token_sequences(os.path.join(txt_folder,"target_valid.txt"),target_dict)

        # #padding sequences
                
        # source_sequence_padded_v = pad_sequences(source_seq_v, maxlen=max_scent_len, padding='post') 
        # target_sequence_padded_v = pad_sequences(target_seq_v, maxlen=max_scent_len, padding='post')         

        # # Converting to tensors
        # # source_sequences_tensor_v = tf.convert_to_tensor(source_sequence_padded_v)
        # # target_sequences_tensor_v = tf.convert_to_tensor(target_sequence_padded_v)

        # #Scentece to tensors test Data 
        # source_seq_te = load_token_sequences(os.path.join(txt_folder,"source_test.txt"),source_dict)

        # target_seq_te = load_token_sequences(os.path.join(txt_folder,"target_test.txt"),target_dict)

        #padding sequences
                
        # source_sequence_padded_te = pad_sequences(source_seq_te, maxlen=max_scent_len, padding='post') 
        # target_sequence_padded_te = pad_sequences(target_seq_te, maxlen=max_scent_len, padding='post')         

        # Converting to tensors
        # source_sequences_tensor_te = tf.convert_to_tensor(source_sequence_padded_te)
        # target_sequences_tensor_te = tf.convert_to_tensor(target_sequence_padded_te)