from datasets import load_dataset
dataset = load_dataset("cfilt/iitb-english-hindi")

source_valid_file = open("./data/txt/source_valid.txt", "w+", encoding='utf8')
target_valid_file = open("./data/txt/target_valid.txt", "w+", encoding='utf8')
for translation_pair in dataset["validation"]["translation"]:
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  source_valid_file.write(source_sentence.strip("\n") + "\n")
  target_valid_file.write(target_sentence.strip("\n") + "\n")
source_valid_file.close()
target_valid_file.close()

source_train_file = open("./data/txt/source_train.txt", "w+", encoding='utf8')
target_train_file = open("./data/txt/target_train.txt", "w+", encoding='utf8')

for translation_pair in dataset["train"]["translation"]:
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  source_train_file.write(source_sentence.strip("\n") + "\n")
  target_train_file.write(target_sentence.strip("\n") + "\n")

source_train_file.close()
target_train_file.close()

source_test_file = open("./data/txt/source_test.txt", "w+", encoding='utf8')
target_test_file = open("./data/txt/target_test.txt", "w+", encoding='utf8')
for translation_pair in dataset["test"]["translation"]:
  source_sentence = translation_pair["en"]
  target_sentence = translation_pair["hi"]
  source_test_file.write(source_sentence.strip("\n") + "\n")
  target_test_file.write(target_sentence.strip("\n") + "\n")
source_test_file.close()
target_test_file.close()