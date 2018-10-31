#----------------------------------- Import Part ------------------------

import tensorflow as tf
import numpy as np
from word_id_test import Word_Id_Map
import sys
print (sys.argv)
import pyttsx3
import os


#------------------------- Text to speeech------------------

engine = pyttsx3.init()
engine.setProperty('rate', 130)

def say(text):
    engine.say(text)
    engine.runAndWait()
#---------------------------------------Model Part-------------------------------
with tf.device('/cpu:0'):
    batch_size = 1
    sequence_length = 25
    num_encoder_symbols = 5004
    num_decoder_symbols = 5004
    embedding_size = 256
    hidden_size = 256
    num_layers = 2

    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        tf.unstack(encoder_inputs, axis=1),
        tf.unstack(decoder_inputs, axis=1),
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        feed_previous=True,
    )
    logits = tf.stack(results, axis=1)
    pred = tf.argmax(logits, axis=2)

    saver = tf.train.Saver()


def check(text):

    try:

        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint('./model/')
            saver.restore(sess, module_file)
            map = Word_Id_Map()
            encoder_input = map.sentence2ids(text)

            encoder_input = encoder_input + [3 for i in range(0, 25 - len(encoder_input))]
            encoder_input = np.asarray([np.asarray(encoder_input)])
            decoder_input = np.zeros([1, 25])
            pred_value = sess.run(pred, feed_dict={encoder_inputs: encoder_input, decoder_inputs: decoder_input})
            sentence = map.ids2sentence(pred_value[0])
            s=""
            x=""
            for i in range(0,len(sentence) ):
                x= sentence[i]
                if  '<pad>'==x:
                    s+=" "
                elif  '<eos>'==sentence[i]+"":
                    s += "."
                elif 'unk'!=sentence[i]:
                    s+=sentence[i]+" "

            print("Tinu ::",s)
            say(s)

    except (KeyError) as e:
        print("Sorry Sir ,  i hear few words for the first time sir :(")



#------------------- Pre Defined Qustion--------------------------
def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data

source_text = load_data('store/pre_qus_ans.txt')
lines=source_text.split("\n")
qus=[]
ans=[]

for i in range(0,lines.__len__()-1):

    if i%2==0 or i==0 :
        qus.append(lines[i].lower())
    if i%2==1:
        ans.append(lines[i].lower())


def checkFormStore(txt):
    for i in range(0, qus.__len__()):
        if txt == qus[i]:
            say(ans[i])
            print("Tinu ::", ans[i])
            return True
    return False

if '__main__' == __name__:
    text = input("you :: ").lower()
    while(text!="exit"):
        t=text.split(" ")
        if(not checkFormStore(text)):
            check(t)
        text = input("You :: ").lower()

