import cv2
import os
import tensorflow as tf

capture = cv2.VideoCapture(0) #-> 0->1 for usb camera
with tf.Session() as sess:
    loader = tf.train.import_meta_graph('./model.meta')
    loader.restore(sess,'./model')
    while(True):
        ret, frame = capture.read()
        show_img = frame.copy()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(show_img, (100,100), (200, 200), (0, 0, 255), 5)
        crop_img = frame[100:200, 100:200]
        cv2.imshow('frame', show_img)
        k = cv2.waitKey(1)
        if k == ord('s'):
            out = sess.run("output:0", feed_dict={"data_in:0": [crop_img]})
            print out 
            cv2.imshow('test', crop_img)

capture.release()
cv2.destroyAllWindows()
