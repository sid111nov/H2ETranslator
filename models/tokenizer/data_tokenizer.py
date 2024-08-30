import os, json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from indicnlp.tokenize import indic_tokenize  
current_dir = os.getcwd()
dict_folder = os.path.join(current_dir,r"data\\dict")
txt_folder = os.path.join(current_dir,r"data\\txt")

# ##
# trying to create new dictionaries to extract new and better dictionary for English to Hindi translations
# this part is still in experimentation phase
# ##

print(txt_folder)

# source_test = []
# with open(os.path.join(txt_folder,"source_test.txt"),'r') as f:
#     for line in f:
        
#         source_test.append(line.rstrip())

target_test = []
with open(os.path.join(txt_folder,"target_test.txt"),'r',encoding='utf-8') as f:
    for line in f:
        
        target_test.append(line.rstrip())
   

# tokenizer_src = tf.keras.preprocessing.text.Tokenizer()

# tokenizer_src.fit_on_texts(source_test)

# vocab_size = len(tokenizer_src.word_index) + 1

# sequences_src = tokenizer_src.texts_to_sequences(source_test)

# max_len = 1000
# padded_sequences_src = tf.keras.preprocessing.sequence.pad_sequences(sequences_src, maxlen=max_len)

# print(padded_sequences_src)

# vocab_src = tokenizer_src.word_index

# with open(os.path.join(dict_folder,"sample_vocab_src.json"),"w") as f:
#     json.dump(vocab_src, f)


# #####

u=0
for text in target_test:
    print(indic_tokenize.trivial_tokenize(text))
    if(u>5):
        break
    u += 1

# tokenized_target_test = [indic_tokenize.trivial_tokenize(text) for text in target_test]

vocab_size_tgt = len(tokenized_target_test) + 1

print(tokenized_target_test) 