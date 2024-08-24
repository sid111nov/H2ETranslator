from subword_nmt import learn_bpe, apply_bpe
import os


#specifying merge operations
merge_operation ="16000"

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir)
txt_dir = os.path.join(root_dir, 'txt')
codes_dir = os.path.join(root_dir, 'codes')

source_bpe_codes = os.path.join(codes_dir,"source-bpe.codes")
target_bpe_codes = os.path.join(codes_dir,"target-bpe.codes")

source_train_input = os.path.join(txt_dir,"source_train.txt")
target_train_input = os.path.join(txt_dir,"target_train.txt")

soruce_full_input = os.path.join(txt_dir,"source_full.txt")
target_full_input = os.path.join(txt_dir,"target_full.txt")

source_test_input = os.path.join(txt_dir,"source_test.txt")
source_valid_input = os.path.join(txt_dir,"source_valid.txt")
target_test_input = os.path.join(txt_dir,"target_test.txt")
target_valid_input = os.path.join(txt_dir,"target_valid.txt")

source_train_output= os.path.join(txt_dir,"source_train_bpe.txt")
source_test_output = os.path.join(txt_dir,"source_test_bpe.txt")
source_valid_output = os.path.join(txt_dir,"source_valid_bpe.txt")
target_train_output = os.path.join(txt_dir,"target_train_bpe.txt")
target_test_output = os.path.join(txt_dir,"target_test_bpe.txt")
target_valid_output = os.path.join(txt_dir,"target_valid_bpe.txt")

source_test_vocab = os.path.join(txt_dir,"source_test_vocab.txt")
source_valid_vocab = os.path.join(txt_dir,"source_valid_vocab.txt")
target_test_vocab = os.path.join(txt_dir,"target_test_vocab.txt")
target_valid_vocab = os.path.join(txt_dir,"target_valid_vocab.txt")
source_train_vocab = os.path.join(txt_dir,"source_train_vocab.txt")
target_train_vocab = os.path.join(txt_dir,"target_train_vocab.txt")


#combining files

os.system(f'type {source_train_input} {source_valid_input} {source_test_input} > {soruce_full_input}')
os.system(f'type {target_train_input} {target_valid_input} {target_test_input} > {target_full_input}')

#learning bpe codes

os.system(f'subword-nmt learn-bpe -s {merge_operation} <{soruce_full_input}> {source_bpe_codes}')  
os.system(f'subword-nmt learn-bpe -s {merge_operation} <{target_full_input}> {target_bpe_codes}')   


#Applying bpe codes
os.system(f'subword-nmt apply-bpe -c {source_bpe_codes} <{source_train_input}> {source_train_output}')
os.system(f'subword-nmt apply-bpe -c {source_bpe_codes} <{source_valid_input}> {source_valid_output}')
os.system(f'subword-nmt apply-bpe -c {source_bpe_codes} <{source_test_input}> {source_test_output}')

os.system(f'subword-nmt apply-bpe -c {target_bpe_codes} <{target_test_input}> {target_test_output}')
os.system(f'subword-nmt apply-bpe -c {target_bpe_codes} <{target_valid_input}> {target_valid_output}')
os.system(f'subword-nmt apply-bpe -c {target_bpe_codes} <{target_train_input}> {target_train_output}')



#Extracting vocabulary
os.system(f'subword-nmt get-vocab <{source_valid_output}> {source_valid_vocab}')
os.system(f'subword-nmt get-vocab <{source_test_output}> {source_test_vocab}')
os.system(f'subword-nmt get-vocab <{target_valid_output}> {target_valid_vocab}')
os.system(f'subword-nmt get-vocab <{target_test_output}> {target_test_vocab}')

os.system(f'subword-nmt get-vocab <{source_train_output}> {source_train_vocab}')
os.system(f'subword-nmt get-vocab <{target_train_output}> {target_train_vocab}')