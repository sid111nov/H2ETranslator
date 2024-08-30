
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

import json
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_tokenization import data_tokenization  
from tensorflow.keras.preprocessing.sequence import pad_sequences

current_dir = os.getcwd()
txt_folder = os.path.join(current_dir,r"data\\txt")
artifact_folder = os.path.join(current_dir,"artifacts")



tokenizer = data_tokenization()

source_dictionary = tokenizer.get_source_dict()
max_index = max(source_dictionary.values())
source_count= len(source_dictionary)

target_dictionary = tokenizer.get_target_dict()
target_count=len(target_dictionary)





#source embedding
model = tf.keras.Sequential()
model.add(Embedding(input_dim=max_index+1, output_dim=128))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(max_index+1,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.01)))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.build(input_shape=(None, tokenizer.max_length()))

model.summary()

max_length = tokenizer.max_length()

valid_src_seq=[]
valid_tgt_seq=[]
test_src_seq=[]
test_tgt_seq=[]

#getting padded sequences
def get_padded_array(seq,maxlength):
    sequence_padded = pad_sequences(seq, maxlen=maxlength, padding='pre') 
    return np.array(sequence_padded)

#converting words to numerical tokens
def get_tokenized_seq(filename,dictionary,lengthf):
    seq=[]
    with open(os.path.join(txt_folder,filename), 'r',encoding='utf-8') as f:
        for line in f:
            seq.append(tokenizer.tokenizer(line,dictionary))
    seq = get_padded_array(seq,lengthf)
    return seq


valid_src_seq = get_tokenized_seq("source_valid.txt",source_dictionary,max_length)
valid_tgt_seq = get_tokenized_seq("target_valid.txt",target_dictionary,max_length)
test_src_seq= get_tokenized_seq("source_test.txt",source_dictionary,max_length)
test_tgt_seq= get_tokenized_seq("target_test.txt",target_dictionary,max_length)


#early stopping criteria definition

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=1, 
    min_delta=0.001,         
    restore_best_weights=True  
)

#pulling data in chunks of lines instead of extracting all scentences
def get_training_chunks():
    with open(os.path.join(txt_folder,"source_train.txt"),'r', encoding='utf-8') as f1,  \
        open(os.path.join(txt_folder,"target_train.txt"),'r', encoding='utf-8')    as f2:
        while True:
            src_chunk = [tokenizer.tokenizer(line.strip(),source_dictionary) for line in (f1.readline() for _ in range(1000)) if line]
            tgt_chunk = [tokenizer.tokenizer(line.strip(),target_dictionary) for line in (f2.readline() for _ in range(1000)) if line]
            src_chunk = get_padded_array(src_chunk,max_length)
            tgt_chunk = get_padded_array(src_chunk,max_length)
            if not src_chunk.any() or not tgt_chunk.any():
                            break
            yield src_chunk, tgt_chunk

#model training
for  src_chunk,  tgt_chunk in    get_training_chunks():  
    if (model.stop_training):
         print("training stopped")
         break    
    model.fit(src_chunk, tgt_chunk, batch_size=10,verbose=1 ,\
            callbacks=[early_stopping], epochs=10, validation_data=(valid_src_seq, valid_tgt_seq))

#evaluating the model
test_loss, test_acc = model.evaluate(test_src_seq, test_tgt_seq)
print(f"Test Accuracy: {test_acc}")

#saving th emodel
model.save(os.path.join(artifact_folder, "translator_model.h5"))




