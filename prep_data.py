import os
import pickle
import copy
import numpy as np
import  tensorflow as tf
import sys
print (sys.argv)



def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data


source_path = 'Txt_data/questions.txt'
target_path = 'Txt_data/answer.txt'
source_text = load_data(source_path)
target_text = load_data(target_path)
# ****************************************************************************************************

import numpy as np
from collections import Counter

english_sentences = source_text.split('\n')
print('* Questions')
print('\t- number of sentences: {}'.format(len(english_sentences)))

french_sentences = target_text.split('\n')
print('* Answers')
print('\t- number of sentences: {} '.format(len(french_sentences)))
print()




source_text_id = []
target_text_id = []

# make a list of sentences (extraction)
source_sentences = source_text.split("\n")
target_sentences = target_text.split("\n")

max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])

# iterating through each sentences (# of sentences in source&target is the same)


#Size Of the  data set
xx=int(input("Full Dataset have 221616 Pairs \n  How many pairs you want to use for training set?? "))


file = open("data.txt", "w")
x=""
y=0



for i in range(0,xx):
    # extract sentences one by one

    source_sentence = source_sentences[i]
    target_sentence = target_sentences[i]

    x = x + source_sentence + "\n"+ target_sentence+ "\n"




file.write(x)
file.close()
print("Total Number Of lines :",xx)
print("Now Run data\data.py \n <<--------------------------- Data Saved in Data.txt ------------------------------------------>>")