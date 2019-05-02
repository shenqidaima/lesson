import tensorflow as tf
import cv2
import glob
import os
import numpy as np

def data_reader(dataset_path):
    data_list = []
    label_list = []
    for cls_path in glob.glob(os.path.join(dataset_path,'*')):
        print cls_path
        for file_name in glob.glob(os.path.join(cls_path,'*')):
            img = cv2.imread(file_name)
            data_list.append(img)
            label_list.append(int(cls_path[-1]))

    data_np = np.array(data_list)
    label_np = np.array(label_list)
    return data_np, label_np

def shuffle_data(data, label):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx,...] , label[idx,...]

def train(data, label):
    data_in = tf.placeholder(tf.float32, [None, 100,100,3], name="data_in")
    print data_in.name
    label_in = tf.placeholder(tf.float32, [None, 3])
    out = tf.layers.conv2d(data_in, 4, 3,padding='same')
    out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
    out = tf.nn.relu(out)
    out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
    out = tf.layers.dense(out, 1000, activation=tf.nn.relu)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)
    pred = tf.layers.dense(out, 3)
    out_label = tf.argmax(pred, 1,name="output")
    print out_label.name

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_in, logits=pred))
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = opt.minimize(loss)
                          
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    batch_size = 16
    for epoch in range(15):
        datas, labels = shuffle_data(data, label)        
        num_batch = len(data)//batch_size
        total_loss = 0
        avg_accuracy = 0
        for batch_idx in range(num_batch):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx+1) * batch_size
            feed_dict = {data_in: datas[start_idx: end_idx, ...],
                         label_in: labels[start_idx: end_idx, ...]}
            correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(label_in,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            _ , loss_out, acc = sess.run([train_op,loss,accuracy], feed_dict = feed_dict)
            total_loss = total_loss + loss_out
            avg_accuracy = avg_accuracy + acc
        print "total loss:", total_loss
        print "avg_accuracy:", avg_accuracy/num_batch
    saver = tf.train.Saver()
    saver.save(sess,'./model')

def dense_to_one_hot(label, num_class):
    num_label = label.shape[0] 
    index_offset = np.arange(num_label) * num_class
    label_one_hot = np.zeros((num_label,num_class))
    label_one_hot.flat[index_offset + label.ravel()] = 1
    return label_one_hot

if __name__ == '__main__':
    dataset_path = "./dataset"
    data, label = data_reader(dataset_path)
    one_hot_label = dense_to_one_hot(label, 3)
    train(data, one_hot_label)
