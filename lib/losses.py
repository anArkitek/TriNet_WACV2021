import keras
import keras.backend as K
import tensorflow as tf


def mae_loss(y_true, y_pred):
    return

def mse_loss(y_true, y_pred):
    print(K.sum((y_true - y_pred) ** 2, axis=-1, keepdims=True).shape)

    return K.sum((y_true - y_pred) ** 2, axis=-1, keepdims=True)

def ortho_loss(y_true, y_pred):

    # v1_preds: (None, 3)
    v1_preds = y_pred[:,  : 3]
    v2_preds = y_pred[:, 3: 6]
    v3_preds = y_pred[:, 6: 9]

    v1_preds = v1_preds / tf.norm(v1_preds, axis=-1, keepdims=True)
    v2_preds = v2_preds / tf.norm(v2_preds, axis=-1, keepdims=True)
    v3_preds = v3_preds / tf.norm(v3_preds, axis=-1, keepdims=True)
    
    v12_cross = K.sum(v1_preds * v2_preds, axis=-1, keepdims=True)
    v13_cross = K.sum(v1_preds * v3_preds, axis=-1, keepdims=True)
    v23_cross = K.sum(v2_preds * v3_preds, axis=-1, keepdims=True)

    return v12_cross**2 + v13_cross**2 + v23_cross**2



def BCE_loss(y_true, y_pred):
    return 0


def loss_sum(y_true, y_pred):

    mse_loss = K.sum((y_true - y_pred) ** 2, axis=-1, keepdims=True)


    v1_preds = y_pred[:,  : 3]
    v2_preds = y_pred[:, 3: 6]
    v3_preds = y_pred[:, 6: 9]

    v1_preds = v1_preds / tf.norm(v1_preds, axis=-1, keepdims=True)
    v2_preds = v2_preds / tf.norm(v2_preds, axis=-1, keepdims=True)
    v3_preds = v3_preds / tf.norm(v3_preds, axis=-1, keepdims=True)
    
    v12_cross = K.sum(K.abs(v1_preds * v2_preds), axis=-1, keepdims=True)
    v13_cross = K.sum(K.abs(v1_preds * v3_preds), axis=-1, keepdims=True)
    v23_cross = K.sum(K.abs(v2_preds * v3_preds), axis=-1, keepdims=True)

    ortho_loss = v12_cross + v13_cross + v23_cross

    print('*'*50)
    print('mse_loss + ortho_loss: ', (mse_loss + ortho_loss).shape)
    print('*'*50)

    return mse_loss + ortho_loss
