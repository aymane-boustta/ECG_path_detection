import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import random
import time
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import tensorflow as tf
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
def plot_roc_auc(y_true, y_pred, class_names, plot_filename):
    roc_curves = {}
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        # Create binary labels for the current class
        y_binary = np.where((y_true == 1) | (y_true == 2), 1, 0)
        fpr, tpr, _ = roc_curve(y_binary, y_pred)
        roc_curves[class_name] = (fpr, tpr)
        auc_score = auc(fpr, tpr)
        auc_scores[class_name] = auc_score
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(auc_score))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Class {}'.format(class_name))
        plt.legend(loc='lower right')
        plt.savefig('{}{}.png'.format(plot_filename, class_name))

    return  auc_scores

    
def plot_confusion_matrix(cm, classes = ['F', 'N', 'S', 'V'], normalize = True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1e-6
        cm = cm.astype('float') / row_sums[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

# @TODO Incroporar medidas de micro Average y micro F1
def evaluate_metrics(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Overlall F1 score
    F1_score = 2 * TP / (2*TP+FP + FN)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    F1_score_mean = np.mean(F1_score)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    return ACC_macro, ACC, TPR, TNR, PPV, F1_score_mean, F1_score
def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size
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

    X_train = X_train[:(X_train.shape[0]/max_time)*max_time,:]
    y_train = y_train[:(X_train.shape[0]/max_time)*max_time]

    X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
    y_train = np.reshape(y_train,[-1,y_test.shape[1]-1,])
    y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
    y_train = np.array(y_train)

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
            class_mask = (y_true >= 1) & (y_true <= 2)
            # This filtering is used to determine how good is the model distinguishing between them
            filtered_y_true = y_true[class_mask]
            filtered_y_pred = y_pred[class_mask]
            sum_test_conf.append(confusion_matrix(y_true, y_pred,labels=range(len(char2numY)-1)))

        sum_test_conf= np.mean(np.array(sum_test_conf, dtype=np.float32), axis=0)
        acc_avg, acc, sensitivity, specificity, PPV, F1_score_mean, F1_score = evaluate_metrics(sum_test_conf)
        print('Average Accuracy is: {:>6.4f}. Averge F1_score is:{:>6.4f} on test set'.format(acc_avg, F1_score_mean))
        for index_ in range(n_classes):
            print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}, F1_score : {:1.4f}".format(classes[index_],
                                                                                                          sensitivity[
                                                                                                              index_],
                                                                                                          specificity[
                                                                                                              index_],PPV[index_],
                                                                                                          acc[index_], F1_score[index_]))
        print("\t Average -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}, F1_score : {:1.4f}".format(np.mean(sensitivity),np.mean(specificity),np.mean(PPV),np.mean(acc),np.mean(F1_score)))
        confusion_matrix_result = confusion_matrix(y_true, y_pred, labels=range(len(char2numY) - 1))
        auc_scores = plot_roc_auc(filtered_y_true, filtered_y_pred, class_names = ['N', 'S'], plot_filename = 'roc_curve_for_N_S')
        # save_confusion_matrix(epochs, confusion_matrix_result, np.mean(acc_track), np.mean(sensitivity),np.mean(specificity), np.mean(PPV), 'confusion_matrix.txt')
        return acc_avg, acc, sensitivity, specificity, PPV, confusion_matrix_result, auc_scores
    loss_track = []
    loss_mean = []
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
            acc_avg, acc, sensitivity, specificity, PPV, confusion_matrix_result, auc_scores = test_model()
        else:

            for epoch_i in range(epochs):
                start_time = time.time()
                train_acc = []
                for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
                    _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                        feed_dict = {inputs: source_batch,
                                     dec_inputs: target_batch[:, :-1],
                                     targets: target_batch[:, 1:]})
                    loss_track.append(batch_loss)
                    train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:,1:])
                accuracy = np.mean(train_acc)
                print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                        accuracy, time.time() - start_time))
                
                loss_mean.append(batch_loss)
                 
                if epoch_i%test_steps==0:
                    acc_avg, acc, sensitivity, specificity, PPV, confusion_matrix_result, auc_scores= test_model()

                    print('loss {:.4f}  ROC score between N and S: {} after {} epochs (batch_size={})'.format(loss_track[-1], auc_scores, epoch_i + 1, batch_size))
                    save_path = os.path.join(checkpoint_dir, ckpt_name)
                    saver.save(sess, save_path)
                    print("Model saved in path: %s" % save_path)
        num_epochs = list(range(1, epochs + 1))   
        plt.figure()
        plot_confusion_matrix(confusion_matrix_result,normalize = True, title='Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        plt.plot(num_epochs, loss_mean, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.grid(True)
        plot_file_path = '/Users/macbookair/Desktop/Thesis_ecg/Database/Test_MITBIH/loss_plot.png'
        try:
            plt.savefig(plot_file_path)
            plt.close()
            print("Loss plot saved to {}".format(plot_file_path))
        except Exception as e:
            print("Error saving loss plot: {}".format(str(e)))
  
        print(str(datetime.now()))
        # test_model()
if __name__ == '__main__':
    main()





