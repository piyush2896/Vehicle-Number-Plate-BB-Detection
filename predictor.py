import tensorflow as tf
import model
import numpy as np
import pandas as pd
from time import time
from data_extractor import extract_data, batches_
import matplotlib.pyplot as plt


def predict(x_input):
    x, y, params = model.generate_model()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/model.ckpt")
        print("\nModel Restored")
        t = time()
        res = sess.run(y, feed_dict={x: x_input})
        print("Prediction Complete\tPrediction Time: {}".format(int(time()-t)))
        sess.close()
    return predict_helper(res)

def predict_helper(res):
    xs = [list(ix) for ix in res[:, range(0, 8, 2)]]
    ys = [list(iy) for iy in res[:, range(1, 8, 2)]]
    [ix.append(ix[0]) for ix in xs]
    [iy.append(iy[0]) for iy in ys]
    return xs, ys


def scale_(xs, ys, x_scaling, y_scaling):
    range_y = [0, 64]
    range_x = [0, 128]

    xs = np.array(xs)
    ys = np.array(ys)

    xs = (xs-range_x[0])/(range_x[1] - range_x[0])
    ys = (ys-range_y[0])/(range_y[1] - range_y[0])

    xs = xs * (x_scaling[1] - x_scaling[0]) + x_scaling[0]
    ys = ys * (y_scaling[1] - y_scaling[0]) + y_scaling[0]
    return xs, ys


if __name__ == '__main__':
    import cv2
    img = cv2.imread('img1.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (128, 64), interpolation=cv2.INTER_AREA)
    img1 = img1.reshape(-1, 64, 128)
    xs, ys = predict(img1)
    xs, ys = scale_(xs, ys, (0, img.shape[1]), (0, img.shape[0]))
    print(xs, ys)
    plt.figure(0)
    plt.imshow(img, cmap='gray')
    plt.plot(xs[0], ys[0], c='r')
    plt.show()