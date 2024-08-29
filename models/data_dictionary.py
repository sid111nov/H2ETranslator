import os


import tensorflow
import json
import keras

from tensorflow.keras.layers import Embedding


current_dir = os.getcwd()
txt_folder = os.path.join(current_dir,r"data\\txt")
dict_folder = os.path.join(current_dir,r"data\\dict")


source_train_vocab = 'source_train_vocab.txt'
target_train_vocab = 'target_train_vocab.txt'


with open(os.path.join(txt_folder,source_train_vocab),'r',encoding='utf-8') as f:
    vocab_src = []
    for line in f:
        vocab_src.append(line.strip().split('@@')[0].split(' ')[0].strip())
    
    

with open(os.path.join(txt_folder,target_train_vocab),'r',encoding='utf-8') as f:
    vocab_tgt = []
    for line in f:
        vocab_tgt.append(line.strip().split('@@')[0].split(' ')[0].strip())
    
vocab_src_dict = {}
vocab_tgt_dict = {}

for i, word in enumerate(vocab_src):
    vocab_src_dict[word]= i

for i, word in enumerate(vocab_tgt):
    vocab_tgt_dict[word]= i    

   
with open(os.path.join(dict_folder,"source_dict.txt"),'w') as f:
    f.write(json.dumps(vocab_src_dict))



with open(os.path.join(dict_folder,"target_dict.txt"),'w', encoding='utf-8') as f:
    f.write(json.dumps(vocab_tgt_dict,ensure_ascii=False))    







