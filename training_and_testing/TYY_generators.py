import keras
import numpy as np
import sys
import tensorflow as tf
import cv2
from tensorflow.python.ops.math_ops import _bucketize as bucketize

def random_crop(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]
    out = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)
    return out

def random_crop_black(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def random_crop_white(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0+255
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def augment_data(images):
    for i in range(0,images.shape[0]):
        
        rand_r = np.random.random()
        if  rand_r < 0.25:
            dn = np.random.randint(15,size=1)[0]+1
            images[i] = random_crop(images[i],dn)

        elif rand_r >= 0.25 and rand_r < 0.5:
            dn = np.random.randint(15,size=1)[0]+1
            images[i] = random_crop_black(images[i],dn)

        elif rand_r >= 0.5 and rand_r < 0.75:
            dn = np.random.randint(15,size=1)[0]+1
            images[i] = random_crop_white(images[i],dn)

        
        if np.random.random() > 0.3:
            images[i] = tf.contrib.keras.preprocessing.image.random_zoom(images[i], [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
        
    return images

def data_generator_pose(X,Y,batch_size):

    while True:
        idxs = np.random.permutation(len(X))
        X_ = X[idxs]
        Y_ = Y[0][idxs]
        p,q = [],[]
        for i in range(len(X_)):
            p.append(X_[i])
            q.append(Y_[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),[np.array(q), np.array(q)]
                p,q = [],[]
        if p:
            yield augment_data(np.array(p)), [np.array(q), np.array(q)]
            p,q = [],[]
