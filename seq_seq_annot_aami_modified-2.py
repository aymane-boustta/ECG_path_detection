import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import random
import time
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score, f1_score,  classification_report, precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import argparse
import itertools

random.seed(654)
def read_mitbih(filename, max_time=100, classes= ['F', 'N', 'S', 'V', 'Q'], max_nlabel=100):
    def normalize(data):
        data = np.nan_to_num(data)  # removing NaNs and Infs
        data = data - np.mean(data)
        data = data / np.std(data)
        return data

    # read data
    data = []  
    samples = spio.loadmat(filename + ".mat")
    samples = samples['s2s_mitbih']
    values = samples[0]['seg_values']
    labels = samples[0]['seg_labels']
    num_annots = sum([item.shape[0] for item in values])

    n_seqs = num_annots / max_time
    #  add all segments(beats) together
    l_data = 0
    for i, item in enumerate(values):
        l = item.shape[0]
        for itm in item:
            if l_data == n_seqs * max_time:
                break
            data.append(itm[0])
            l_data = l_data + 1

    #  add all labels together
    l_labels  = 0
    t_labels = []
    for i, item in enumerate(labels):
        if len(t_labels)==n_seqs*max_time:
            break
        item= item[0]
        for lebel in item:
            if l_labels == n_seqs * max_time:
                break
            t_labels.append(str(lebel))
            l_labels = l_labels + 1

    del values
    data = np.asarray(data)
    shape_v = data.shape
    data = np.reshape(data, [shape_v[0], -1])
    t_labels = np.array(t_labels)
    _data  = np.asarray([],dtype=np.float64).reshape(0,shape_v[1])
    _labels = np.asarray([],dtype=np.dtype('|S1')).reshape(0,)
    for cl in classes:
        _label = np.where(t_labels == cl)
        permute = np.random.permutation(len(_label[0]))
        _label = _label[0][permute[:max_nlabel]]
        _data = np.concatenate((_data, data[_label]))
        _labels = np.concatenate((_labels, t_labels[_label]))

    data = _data[:(len(_data)/ max_time) * max_time, :]
    _labels = _labels[:(len(_data) / max_time) * max_time]

    # data = _data
    #  split data into sublist of 100=se_len values
    data = [data[i:i + max_time] for i in range(0, len(data), max_time)]
    labels = [_labels[i:i + max_time] for i in range(0, len(_labels), max_time)]
    # shuffle
    permute = np.random.permutation(len(labels))
    data = np.asarray(data)
    labels = np.asarray(labels)
    data= data[permute]
    labels = labels[permute]
    print('Records processed!')
    return data, labels



def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size
# def batch_data(x, y, batch_size):
#     shuffle = np.random.permutation(len(x))
#     start = 0
#     x = x[shuffle]
#     y = y[shuffle]
#     max_seq_length = max(max(len(seq) for seq in batch_x) for batch_x in x)
#     while start + batch_size <= len(x):
#         batch_x, batch_y = x[start:start + batch_size], y[start:start + batch_size]

#         # Dynamically calculate max sequence length in the current batch
#         max_seq_length = max(max_seq_length, max(max(len(seq) for seq in batch_x), max(len(seq) for seq in batch_y)))

#         # Pad sequences to the same length
#         padded_batch_x = pad_sequences(batch_x, max_seq_length)
#         padded_batch_y = pad_sequences(batch_y, max_seq_length)  # Target sequences have the same length

#         yield padded_batch_x, padded_batch_y
#         start += batch_size

    
def build_network(inputs, dec_inputs,char2numY,n_channels=10,input_depth=280,num_units=128,max_time=10,bidirectional=False):
    _inputs = tf.reshape(inputs, [-1, n_channels, input_depth / n_channels])

    # #(batch*max_time, 280, 1) --> (N, 280, 18)
    conv1 = tf.layers.conv1d(inputs=_inputs, filters=32, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=128, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)

    shape = conv3.get_shape().as_list()
    data_input_embed = tf.reshape(conv3, (-1, max_time, shape[1] * shape[2]))
    embed_size = 10  # 128 lstm_size # shape[1]*shape[2]

    # Embedding layers
    output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
    data_output_embed = tf.nn.embedding_lookup(output_embedding, dec_inputs)

    with tf.variable_scope("encoding") as encoding_scope:
        if not bidirectional:

            # Regular approach with LSTM units
            lstm_enc = tf.contrib.rnn.LSTMCell(num_units)
            _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=data_input_embed, dtype=tf.float32)

        else:

            # Using a bidirectional LSTM architecture instead
            enc_fw_cell = tf.contrib.rnn.LSTMCell(num_units)
            enc_bw_cell = tf.contrib.rnn.LSTMCell(num_units)

            ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=enc_fw_cell,
                cell_bw=enc_bw_cell,
                inputs=data_input_embed,
                dtype=tf.float32)
            enc_fin_c = tf.concat((enc_fw_final.c, enc_bw_final.c), 1)
            enc_fin_h = tf.concat((enc_fw_final.h, enc_bw_final.h), 1)
            last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)

    with tf.variable_scope("decoding") as decoding_scope:
        if not bidirectional:
            lstm_dec = tf.contrib.rnn.LSTMCell(num_units)
        else:
            lstm_dec = tf.contrib.rnn.LSTMCell(2 * num_units)

        dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=data_output_embed, initial_state=last_state)

    logits = tf.layers.dense(dec_outputs, units=len(char2numY), use_bias=True)

    return logits
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_time', type=int, default=10)
    parser.add_argument('--test_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='Test_MITBIH/s2s_mitbih_aami')
    parser.add_argument('--bidirectional', type=str2bool, default=str2bool('False'))
    # parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--num_units', type=int, default=128)
    parser.add_argument('--n_oversampling', type=int, default=10000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints-seq2seq')
    parser.add_argument('--ckpt_name', type=str, default='seq2seq_mitbih.ckpt')
    parser.add_argument('--classes', nargs='+', type=chr,
                        default=['F','N', 'S','V'])
    args = parser.parse_args()
    run_program(args)
def run_program(args):
    print(args)
    max_time = args.max_time # 5 3 second best 10# 40 # 100
    epochs = args.epochs # 300
    batch_size = args.batch_size # 10
    num_units = args.num_units
    bidirectional = args.bidirectional
    n_oversampling = args.n_oversampling
    checkpoint_dir = args.checkpoint_dir
    ckpt_name = args.ckpt_name
    test_steps = args.test_steps
    classes= args.classes
    filename = args.data_dir

    X, Y = read_mitbih(filename,max_time,classes=classes,max_nlabel=100000) #11000
    print ("# of sequences: ", len(X))
    input_depth = X.shape[2]
    n_channels = 10
    classes = np.unique(Y)
    char2numY = dict(zip(classes, range(len(classes))))
    n_classes = len(classes)
    print ('Classes: ', classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print (cl, len(np.where(Y.flatten()==cl)[0]))

    char2numY['<GO>'] = len(char2numY)
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))

    Y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in Y]
    Y = np.array(Y)

    y_seq_length = len(Y[0])- 1
    
    # Placeholders
    inputs = tf.placeholder(tf.float32, [None, max_time, input_depth], name = 'inputs')
    targets = tf.placeholder(tf.int32, (None, None), 'targets')
    dec_inputs = tf.placeholder(tf.int32, (None, None), 'output')

    logits = build_network(inputs, dec_inputs, char2numY, n_channels=n_channels, input_depth=input_depth, num_units=num_units, max_time=max_time,
                  bidirectional=bidirectional)
    with tf.name_scope("optimization"):
        # Loss function
        vars = tf.trainable_variables()
        beta = 0.001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                            if 'bias' not in v.name]) * beta
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
        #loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights=tf.ones([batch_size, tf.shape(targets)[1]]), average_across_timesteps=True, average_across_batch=True)
        # Optimizer
        loss = tf.reduce_mean(loss + lossL2)
        optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)


    # split the dataset into the training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # over-sampling: SMOTE
    X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
    y_train= y_train[:,1:].flatten()

    nums = []
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        nums.append(len(np.where(y_train.flatten()==ind)[0]))
    # ratio={0:nums[3],1:nums[1],2:nums[3],3:nums[3]} # the best with 11000 for N
    ratio={0:n_oversampling,1:nums[1],2:n_oversampling,3:n_oversampling}
    sm = SMOTE(random_state=12,ratio=ratio)
    X_train, y_train = sm.fit_sample(X_train, y_train)

    # X_train = X_train[:(X_train.shape[0]/max_time)*max_time,:]
    # y_train = y_train[:(X_train.shape[0]/max_time)*max_time]

    # X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
    # y_train = np.reshape(y_train,[-1,y_test.shape[1]-1,])
    
    X_train = X_train[:(X_train.shape[0]//max_time)*max_time, :]
    y_train = y_train[:(X_train.shape[0]//max_time)*max_time]

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1] - 1])

    y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
    y_train = np.array(y_train)


    print("Length of X_train:", len(X_train))
    print("Length of y_train:", len(y_train))
    print("Length of X_test:", len(X_test))
    print("Length of y_test:", len(y_test))


    print ('Classes in the training set: ', classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print (cl, len(np.where(y_train.flatten()==ind)[0]))
    print ("------------------y_train samples--------------------")
    for ii in range(2):
      print(''.join([num2charY[y_] for y_ in list(y_train[ii+5])]))
    print ("------------------y_test samples--------------------")
    for ii in range(2):
      print(''.join([num2charY[y_] for y_ in list(y_test[ii+5])]))

    def test_model():
        acc_track = []
        sum_test_conf = []
        y_true_batch = []  
        y_pred_batch = []
        for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_test, y_test, batch_size)):

            dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
            for i in range(y_seq_length):
                batch_logits = sess.run(logits,
                                        feed_dict={inputs: source_batch, dec_inputs: dec_input})
                prediction = batch_logits[:, -1].argmax(axis=-1)
                dec_input = np.hstack([dec_input, prediction[:, None]])

            acc_track.append(dec_input[:, 1:] == target_batch[:, 1:])
            y_true= target_batch[:, 1:].flatten()
            y_pred = dec_input[:, 1:].flatten()
            y_true_batch.extend(y_true)  
            y_pred_batch.extend(y_pred)
            
        print("The shape of y_pred : " + str(len(y_pred_batch))) # 200 pulses x 101 the length of the sequence y
        print("The shape of y_true  : " + str(len(y_true_batch)))
        confusion_matrix_result = confusion_matrix(y_true_batch, y_pred_batch, labels=range(len(char2numY) - 1))
        print("------------------------- Confusion matrix ------------------------")
        print(confusion_matrix_result)
        classification_report_result = classification_report(y_true_batch, y_pred_batch, target_names=classes, digits=4)

        
        return confusion_matrix_result, classification_report_result
    train_loss_history = []
    test_loss_history = []
    loss_track = []
    def count_prameters():
        print ('# of Params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    count_prameters()

    if (os.path.exists(checkpoint_dir) == False):
        os.mkdir(checkpoint_dir)
    # train the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        print(str(datetime.now()))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        pre_acc_avg = 0.0
        if ckpt and ckpt.model_checkpoint_path:
            # # Restore
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            confusion_matrix_result, classification_report_result = test_model()
        else:
            
            #max_seq_length = max(max(len(seq) for seq in seqs) for seqs in X_train)
            for epoch_i in range(epochs):
                train_loss = 0.0
                start_time = time.time()
                train_acc = []
                
                for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
                    

                    _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                        feed_dict = {inputs: source_batch,
                                     dec_inputs: target_batch[:, :-1],
                                     targets: target_batch[:, 1:]})
                    
                    loss_track.append(batch_loss)
                    train_loss += batch_loss
                    train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:,1:])
                    
                train_loss /= (batch_i + 1)
                accuracy = np.mean(train_acc)
                train_loss_history.append(train_loss)
                print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                        accuracy, time.time() - start_time))
                
                #if epoch_i%test_steps==0 or epoch_i==epochs - 1:
                if epoch_i % 3 == 0 or epoch_i==epochs - 1:
                    print("------------------------------- Testing the model --------------------------------")
                    test_loss = 0.0  
                    test_acc = []

                    for batch_i, (test_source_batch, test_target_batch) in enumerate(batch_data(X_test, y_test, batch_size)):
                        # Calculate the loss for the test data
                        batch_loss, _ = sess.run([loss, logits], feed_dict={
                        inputs: test_source_batch,
                        dec_inputs: test_target_batch[:, :-1],
                        targets: test_target_batch[:, 1:]
                        })
                        test_loss += batch_loss

                    test_loss /= (batch_i + 1)  # Calculate the average test loss
                    print('Epoch {} Test Loss: {:.4f}'.format(epoch_i, test_loss))
                    test_loss_history.append(test_loss)
                    confusion_matrix_result, classification_report_result = test_model()
                    print(classification_report_result)
                    print("----------------------------------------------------------------------------------")            

            #plot_loss(loss_track, test_loss_history)
            plt.figure()
            train_epochs = range(0, epochs)
            test_epochs = range(0, epochs, max(1, epochs // len(test_loss_history)))[:len(test_loss_history)]
            plt.plot(train_epochs, train_loss_history, label='Training Loss', marker='o')
            plt.plot(test_epochs, test_loss_history, label='Testing Loss', marker='x')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Testing Loss vs. Epochs')
            plt.legend()
            plot_filename = "loss_plot.png"
            plt.savefig(plot_filename)
            save_path = os.path.join(checkpoint_dir, ckpt_name)
            saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)
            
            
  
        print(str(datetime.now()))
        # test_model()
if __name__ == '__main__':
    main()





