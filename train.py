import tensorflow as tf
import numpy as np
import time


def loadQA():
    train_x = np.load('./data/idx_q.npy', mmap_mode='r')
    train_y = np.load('./data/idx_a.npy', mmap_mode='r')
    train_target = np.load('./data/idx_o.npy', mmap_mode='r')

    return train_x, train_y, train_target

#--------------------------- Peramiter------------
batch_size =800
sequence_length = 25
hidden_size = 256
num_layers = 2
num_encoder_symbols = 5004  # 'UNK' and '<go>' and '<eos>' and '<pad>'   vocab size + xtra words
num_decoder_symbols = 5004
embedding_size = 256
learning_rate = 0.0001
model_dir = './model'
epoch = 0
epch = 1000
cost = 5
loop_limit = 0.001

# Creates a graph.
with tf.device('/device:GPU:0'):
    #Built a Model
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
        feed_previous=False
    )
    logits = tf.stack(results, axis=1)
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets=targets, weights=weights)
    pred = tf.argmax(logits, axis=2)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    saver = tf.train.Saver()
    train_weights = np.ones(shape=[batch_size, sequence_length], dtype=np.float32)


    #Training Part ----------->>

    sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True))
    print("Be Patient Bcz Training Start :D :D")

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Old Model Restore.....")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Create New Model..")
        sess.run(tf.global_variables_initializer())

    old_cost=0.2
    f = open("log.txt", "w")
    f.close()
    while cost > loop_limit:

        epoch = epoch + 1
        train_x, train_y, train_target = loadQA()
        a = int(len(train_x) // batch_size)



        print("Epoch ------------------------------------------------------------->>>", epoch)
        str=time.time()

        for step in range(1, a):


            print("||{}".format("."*step))
            train_x, train_y, train_target = loadQA()
            train_encoder_inputs = train_x[step * batch_size:step * batch_size + batch_size, :]
            train_decoder_inputs = train_y[step * batch_size:step * batch_size + batch_size, :]
            train_targets = train_target[step * batch_size:step * batch_size + batch_size, :]
            op = sess.run(train_op, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                               weights: train_weights, decoder_inputs: train_decoder_inputs})
            cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                             weights: train_weights, decoder_inputs: train_decoder_inputs})

            step = step + 1

        stp=time.time()
        diff=stp-str
        cost_diff=abs(old_cost-cost)
        old_cost=cost
        print("Cost :: ",cost)
        print("Remaining Steps: {} ".format(abs(int((loop_limit - cost) / cost_diff))))


        f = open("log.txt", "a")
        f.write("Step---------------:: {} \n Cost:: {} \n\n".format(epoch,cost))
        f.close()
        if epoch % 2 == 0:
            saver.save(sess, model_dir + '/model.ckpt', global_step=epoch + 1)
            print("<---------------------------- Model Saved ---------------------------->>")


        print("<<-------------------------- End Of Step {} ----------------------------->>".format(epoch))
