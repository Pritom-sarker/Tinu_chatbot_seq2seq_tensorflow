import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
tf.__version__

# Load the data
lines = open('raw_data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('raw_data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))



# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))


# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []


file = open("Txt_data/qus&ans.txt", "a")
q="".join(id2line)
file.write(q)
file.close()


for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

# Compare lengths of questions and answers
print("Question Len:",len(questions))
print("Answer Len:",len(answers))



def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    text.replace("  ", " ")
    text.replace("   ", " ")
    text.replace("    ", " ")
    text.replace("     ", " ")
    text.replace("      ", " ")
    text.replace("       ", " ")
    text.replace("        ", " ")
    text.replace("         ", " ")
    text.replace("          ", " ")
    text.replace("           ", " ")
    text.replace("brandon\ttime", " ")
    text = re.sub(r"brandon time", "brandon", text)
    text = re.sub(r"night	159", "night", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"   ", " ", text)
    text = re.sub(r"    ", " ", text)
    text = re.sub(r"     ", " ", text)
    text = re.sub(r"      ", " ", text)
    text = re.sub(r"       ", " ", text)


    text = re.sub(r"159	nt", "", text)
    return text



# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

q=""
for i in range(0,len(clean_questions)):
   q=clean_questions[i].join("\n").join(clean_answers[i])


file = open("Txt_data/questions.txt", "w")
q="\n".join(clean_questions)
file.write(q)
file.close()


file = open("Txt_data/answer.txt", "w")
q="\n".join(clean_answers)
file.write(q)
file.close()

print("Now Run prep_data.py \n<<--------------- All Question & Answers is  Saved On Txt_data Folder  ------------------------------->>")