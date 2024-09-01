import os, json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from indicnlp.tokenize import indic_tokenize  
current_dir = os.getcwd()
dict_folder = os.path.join(current_dir,"data","dict")
txt_folder = os.path.join(current_dir,"data","txt")
from collections import Counter

# ##
# trying to create new dictionaries to extract new and better dictionary for English to Hindi translations
# this part is still in experimentation phase
# ##


source_test = []
with open(os.path.join(txt_folder,"source_train.txt"),'r',encoding='utf-8') as f:
    for line in f:
        
        source_test.append(line.rstrip())

target_test = []
with open(os.path.join(txt_folder,"target_train.txt"),'r',encoding='utf-8') as f:
    for line in f:
        
        target_test.append(line.rstrip())
   

tokenizer_src = tf.keras.preprocessing.text.Tokenizer()

tokenizer_src.fit_on_texts(source_test)

vocab_size = len(tokenizer_src.word_index) + 1

sequences_src = tokenizer_src.texts_to_sequences(source_test)


vocab_src = tokenizer_src.word_index

with open(os.path.join(dict_folder,"english_dictonary.txt"),"w",encoding='utf-8') as f:
    f.write(json.dumps(vocab_src))


# #####



tokenized_target_test = [indic_tokenize.trivial_tokenize(text) for text in target_test]

#vocab_size_tgt = len(tokenized_target_test) + 1

#print(tokenized_target_test) 

hindi_vocab=[]
hindi_dictionary={}

# with open(os.path.join(dict_folder,"hindi_words.txt"),"w", encoding='utf-8') as f:
for tokens in tokenized_target_test:
    for token in tokens:
        hindi_vocab.append(token)

word_list = Counter(hindi_vocab)    

sorted_freq = word_list.most_common()

#print(sorted_freq)

filtered_word_freq = [word for word, freq in sorted_freq if freq >= 15]



for i,word in enumerate(filtered_word_freq):
    hindi_dictionary[word]=i

print(max(hindi_dictionary.values()), len(filtered_word_freq))

# with open(os.path.join(dict_folder,"hindi_freq.txt"),'w',encoding='utf-8') as f:
#     f.write(json.dumps(sorted_freq,ensure_ascii=False))


with open(os.path.join(dict_folder,"hindi_dictonary.txt"),'w',encoding='utf-8') as f:
    f.write(json.dumps(hindi_dictionary,ensure_ascii=False))