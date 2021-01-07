import sys
import logging
import numpy as np

import tensorflow as tf

from keras.models import Model
from keras.applications.resnet50 import ResNet50

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Layer
from keras.layers import Reshape
from keras.layers import Multiply
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization


from keras import backend as K

from .capsulelayers import CapsuleLayer
from .capsulelayers import MatMulLayer

from .utils import register_keras_custom_object

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)

# Custom layers
# Note - Usage of Lambda layers prevent the convertion
# and the optimizations by the underlying math engine (tensorflow in this case)

@register_keras_custom_object
class SSRLayer(Layer):
    def __init__(self, s1, s2, s3, lambda_d, **kwargs):
        super(SSRLayer, self).__init__(**kwargs)
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.lambda_d = lambda_d
        self.trainable = False

    def call(self, inputs):
        #inputs shape: (?,3,39)
        x = inputs
        a = x[:, :, 0] * 0
        b = x[:, :, 0] * 0
        c = x[:, :, 0] * 0
        
        s1 = 3
        s2 = 9
        s3 = 27
    
        di = s1 // 2    
        dj = s2 // 2    
        dk = s3 // 2
        
        V = 1

        #s1 = 3
        # i = 0, 1, 2 ~> i-di = -1, 0, 1
        for i in range(0, s1):
            a = a + (i - di) * x[:, :, i]
        a = a / (s1//2)
        
        #s2 = 9
        # j - dj ~> [-4, 4]
        for j in range(0, s2):
            b = b + (j - dj) * x[:, :, j+3]
        b = b / (s2//2)
        
        
        #s3 = 27
        for k in range(0, s3):
            c = c + (k - dk) * x[:, :, k+12]
        c = c / (s3//2)
        
        pred = (a+b+c) / 3


        return pred

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3)

    def get_config(self):
        config = {
            's1': self.s1,
            's2': self.s2,
            's3': self.s3,
            'lambda_d': self.lambda_d
        }
        base_config = super(SSRLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



@register_keras_custom_object
class FeatSliceLayer(Layer):
    def __init__(self, start_index, end_index,  **kwargs):
        super(FeatSliceLayer, self).__init__(**kwargs)
        self.start_index = start_index
        self.end_index = end_index
        self.trainable = False

    def call(self, inputs):    
        return inputs[:,self.start_index:self.end_index]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.end_index - self.start_index)

    def get_config(self):
        config = {
            'start_index': self.start_index,
            'end_index': self.end_index
        }
        base_config = super(FeatSliceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@register_keras_custom_object
class MomentsLayer(Layer):
    def __init__(self, **kwargs):
        super(MomentsLayer,self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):        
        _, var = tf.nn.moments(inputs,axes=-1)
        #var    : (batch_size, feature_map_width, feature_map_height)
        return var

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

@register_keras_custom_object
class MatrixMultiplyLayer(Layer):
    def __init__(self, **kwargs):
        super(MatrixMultiplyLayer,self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):                
        x1, x2 = inputs
        # TODO: add some asserts on the inputs
        # it is expected the shape of inputs are 
        # arranged to be able to perform the matrix multiplication
        return tf.matmul(x1,x2)

    def compute_output_shape(self, input_shapes):        
        return (input_shapes[0][0],input_shapes[0][1], input_shapes[1][-1])

@register_keras_custom_object
class MatrixNormLayer(Layer):
    def __init__(self, tile_count,  **kwargs):
        super(MatrixNormLayer,self).__init__(**kwargs)
        self.trainable = False
        self.tile_count = tile_count

    def call(self, input):                
        sum = K.sum(input,axis=-1,keepdims=True)        
        tiled = K.tile(sum,(1,1,self.tile_count))        
        return tiled

    def compute_output_shape(self, input_shape):        
        return (input_shape[0], input_shape[1], self.tile_count)

    def get_config(self):
        config = {
            'tile_count': self.tile_count
        }
        base_config = super(MatrixNormLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@register_keras_custom_object
class PrimCapsLayer(Layer):
    def __init__(self, **kwargs):
        super(PrimCapsLayer,self).__init__(**kwargs)
        self.trainable = False        

    def call(self, inputs):                
        x1, x2, norm = inputs
        return tf.matmul(x1,x2) / norm

    def compute_output_shape(self, input_shapes):                
        return input_shapes[-1]

@register_keras_custom_object
class AggregatedFeatureExtractionLayer(Layer):
    def __init__(self, num_capsule,  **kwargs):
        super(AggregatedFeatureExtractionLayer,self).__init__(**kwargs)
        self.trainable = False
        self.num_capsule = num_capsule

    def call(self, input):                
        s1_a = 0
        s1_b = self.num_capsule//3
        # input[:, 0: 1, :]
        feat_s1_div = input[:,s1_a:s1_b,:]
        s2_a = self.num_capsule//3
        s2_b = 2*self.num_capsule//3
        # input[:, 1: 2, :]
        feat_s2_div = input[:,s2_a:s2_b,:]
        s3_a = 2*self.num_capsule//3
        s3_b = self.num_capsule
        # input[:, 2: 3, :]
        feat_s3_div = input[:,s3_a:s3_b,:]

        return [feat_s1_div, feat_s2_div, feat_s3_div]

    def compute_output_shape(self, input_shape):        
        last_dim = input_shape[-1]
        partition = self.num_capsule//3
        return [(input_shape[0], partition, last_dim), (input_shape[0], partition, last_dim), (input_shape[0], partition, last_dim)]

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule
        }
        base_config = super(AggregatedFeatureExtractionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BaseFSANet(object):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        '''
        Args:
            image_size  : 64;
            num_classes : 3;            roll, pitch, yaw
            stage_num   : [3, 3, 3];    # of bins in each stage
            lambda_d    : 1.0;          Control Delta
            S_set       : [];           Parameters of Capsules
        '''
        
        self._channel_axis = 3 if K.image_data_format() == 'channels_last' else 1

        if self._channel_axis == 1:
            logging.debug("image_dim_ordering = 'th'")
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._input_shape = (image_size, image_size, 3)


        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        ''''
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3 or 8*8*3
        m_dim = 5
        '''
        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        # ? F_shape = 16
        self.F_shape = int(self.num_capsule / 3) * self.dim_capsule
        # ? map_xy_size = 8
        self.map_xy_size = int(8 * image_size / 64)

        # is_fc_model
        self.is_fc_model = False
        self.is_noS_model = False
        self.is_varS_model = False

    def ssr_build_resnet(self, input_size):
        resnet = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=input_size, pooling=None, classes=1000)

        model = Model(inputs=resnet.input, outputs=[AveragePooling2D((2,2))(Conv2D(64,(1,1))(resnet.get_layer('activation_10').output)),
                                        Conv2D(64,(1,1))(resnet.get_layer('activation_16').output),
                                        Conv2D(64,(1,1))(resnet.get_layer('activation_22').output)], name='ssr_backbone')

        return model


    def _convBlock(self, x, num_filters, activation, kernel_size=(3,3)):
        x = SeparableConv2D(num_filters,kernel_size,padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        return x

    def ssr_F_model_build(self, feat_dim, name_F, vec_order):
        input_s1_pre = Input((feat_dim,))
        input_s2_pre = Input((feat_dim,))
        input_s3_pre = Input((feat_dim,))

        def _process_input(stage_index, stage_num, num_classes, input_s_pre):
            # input_s_pre   : (None, 16)
            bins_num = stage_num ** stage_index
            units_num = 3 * bins_num
            assert units_num in [9, 27, 81]
            prob_all_bins = Reshape((3, bins_num))(Dense(units=units_num, 
                                                          activation='sigmoid',
                                                          name='all_bins_{}'.format(stage_index))(input_s_pre))
            
            # delta_s   : (None, 3)
            # local_s   : (None, 3)
            # pred_s    : (None, 3, 3)
            # return delta_s, local_s, pred_s

            return prob_all_bins
        ###########################################################################################

        # delta_s1  : (None, 3)
        # local_s1  : (None, 3)
        # pred_s1   : (None, 3, 3)

        # prob_s1: [None, 3, 3]
        # prob_s2: [None, 3, 9]
        # prob_s3: [None, 3, 27]
        prob_s1  = _process_input(1, self.stage_num[0], self.num_classes, input_s1_pre)
        prob_s2  = _process_input(2, self.stage_num[1], self.num_classes, input_s2_pre)
        prob_s3  = _process_input(3, self.stage_num[2], self.num_classes, input_s3_pre)

        # prob_merge: (None, 3, 39)
        prob_merge = Concatenate(axis=-1)([prob_s1, prob_s2, prob_s3])

        # return Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F + f'_{vec_order}')
        return Model(inputs=[input_s1_pre, input_s2_pre, input_s3_pre],
                     outputs=prob_merge,
                     name=name_F + '_{}'.format(vec_order))


    def ssr_FC_model_build(self, feat_dim, name_F):
        input_s1_pre = Input((feat_dim,))
        input_s2_pre = Input((feat_dim,))
        input_s3_pre = Input((feat_dim,))

        def _process_input(stage_index, stage_num, num_classes, input_s_pre):
            feat_delta_s = Dense(2 * num_classes, activation='tanh')(input_s_pre)
            delta_s = Dense(num_classes, activation='tanh', name='delta_s{}'.format(stage_index))(feat_delta_s)

            feat_local_s = Dense(2 * num_classes, activation='tanh')(input_s_pre)
            local_s = Dense(units=num_classes, activation='tanh', name='local_delta_stage{}'.format(stage_index))(feat_local_s)

            feat_pred_s = Dense(stage_num * num_classes,activation='relu')(input_s_pre) 
            pred_s = Reshape((num_classes,stage_num))(feat_pred_s)

            return delta_s, local_s, pred_s

        delta_s1, local_s1, pred_s1 = _process_input(1, self.stage_num[0], self.num_classes, input_s1_pre)
        delta_s2, local_s2, pred_s2 = _process_input(2, self.stage_num[1], self.num_classes, input_s2_pre)
        delta_s3, local_s3, pred_s3 = _process_input(3, self.stage_num[2], self.num_classes, input_s3_pre)        
           
        return Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)


    def ssr_feat_S_model_build(self, m_dim):
        
        input_preS = Input((self.map_xy_size,self.map_xy_size,64))        

        # is_varS_model compute teh variance
        if self.is_varS_model:
            feat_preS = MomentsLayer()(input_preS)
        else:
            feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)        

        feat_preS = Reshape((-1,))(feat_preS)

        SR_matrix = Dense(m_dim*(self.map_xy_size*self.map_xy_size*3),activation='sigmoid')(feat_preS)
        SR_matrix = Reshape((m_dim,(self.map_xy_size*self.map_xy_size*3)))(SR_matrix)
        
        return Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')

    def ssr_S_model_build(self, num_primcaps, m_dim, vec_order):
        # Input: (8, 8, 64)
        # s1: means stage 1?
        input_s1_preS = Input((self.map_xy_size,self.map_xy_size,64))
        input_s2_preS = Input((self.map_xy_size,self.map_xy_size,64))
        input_s3_preS = Input((self.map_xy_size,self.map_xy_size,64))

        # 这里有两种选择：
        # 根据 is_varS_model 来判断是否计算 variance
        feat_S_model = self.ssr_feat_S_model_build(m_dim)

        SR_matrix_s1,feat_s1_preS = feat_S_model(input_s1_preS)
        SR_matrix_s2,feat_s2_preS = feat_S_model(input_s2_preS)
        SR_matrix_s3,feat_s3_preS = feat_S_model(input_s3_preS)
        
        # by default, axis=-1
        # keep the size of the feature map the same, concatenate the channels
        feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
        
        # int(num_primcaps / 3) == 7 or 8*8
        # m_dim == 5
        SL_matrix = Dense(int(num_primcaps / 3) * m_dim,activation='sigmoid')(feat_pre_concat)
        SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)       

        S_matrix_s1 = MatrixMultiplyLayer(name="S_matrix_s1")([SL_matrix,SR_matrix_s1])
        S_matrix_s2 = MatrixMultiplyLayer(name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
        S_matrix_s3 = MatrixMultiplyLayer(name='S_matrix_s3')([SL_matrix,SR_matrix_s3])        

        # Very important!!! Without this training won't converge.        
        # norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
        norm_S_s1 = MatrixNormLayer(tile_count=64)(S_matrix_s1)
        norm_S_s2 = MatrixNormLayer(tile_count=64)(S_matrix_s2)
        norm_S_s3 = MatrixNormLayer(tile_count=64)(S_matrix_s3)        

        # map_xy_size == 8
        # feat_sk_pre   : (8, 8, 64)
        feat_s1_pre = Reshape((self.map_xy_size*self.map_xy_size,64))(input_s1_preS)
        feat_s2_pre = Reshape((self.map_xy_size*self.map_xy_size,64))(input_s2_preS)
        feat_s3_pre = Reshape((self.map_xy_size*self.map_xy_size,64))(input_s3_preS)

        # feat_pre_concat   : (8, 24, 64)
        feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
        
        # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
        # https://github.com/keras-team/keras/issues/9779
        # Make sure 'tf.matmul' is used
        # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
        primcaps_s1 = PrimCapsLayer()([S_matrix_s1,feat_pre_concat, norm_S_s1])
        primcaps_s2 = PrimCapsLayer()([S_matrix_s2,feat_pre_concat, norm_S_s2])
        primcaps_s3 = PrimCapsLayer()([S_matrix_s3,feat_pre_concat, norm_S_s3])        
        
        primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

        return Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model_{}'.format(vec_order))

    def ssr_noS_model_build(self, vec_order, **kwargs):        

        input_s1_preS = Input((self.map_xy_size,self.map_xy_size,64))
        input_s2_preS = Input((self.map_xy_size,self.map_xy_size,64))
        input_s3_preS = Input((self.map_xy_size,self.map_xy_size,64))

        primcaps_s1 = Reshape((self.map_xy_size*self.map_xy_size,64))(input_s1_preS)
        primcaps_s2 = Reshape((self.map_xy_size*self.map_xy_size,64))(input_s2_preS)
        primcaps_s3 = Reshape((self.map_xy_size*self.map_xy_size,64))(input_s3_preS)

        primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])
        print(vec_order)
        return Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model_{}'.format(vec_order))

    def __call__(self):
        logging.debug("Creating model...")
        img_inputs = Input(self._input_shape)        

        # Build various models
        # Two-stream structure for extracting the features.
        ssr_G_model = self.ssr_build_resnet(self._input_shape)
        
        if self.is_noS_model:
            ssr_S_model_0 = self.ssr_noS_model_build(vec_order=0)
            ssr_S_model_1 = self.ssr_noS_model_build(vec_order=1)
            ssr_S_model_2 = self.ssr_noS_model_build(vec_order=2)

        else:
            ssr_S_model_0 = self.ssr_S_model_build(num_primcaps=self.num_primcaps,m_dim=self.m_dim, vec_order=0)
            ssr_S_model_1 = self.ssr_S_model_build(num_primcaps=self.num_primcaps,m_dim=self.m_dim, vec_order=1)
            ssr_S_model_2 = self.ssr_S_model_build(num_primcaps=self.num_primcaps,m_dim=self.m_dim, vec_order=2)

        ssr_aggregation_model_0 = self.ssr_aggregation_model_build((self.num_primcaps,64), vec_order=0)
        ssr_aggregation_model_1 = self.ssr_aggregation_model_build((self.num_primcaps,64), vec_order=1)
        ssr_aggregation_model_2 = self.ssr_aggregation_model_build((self.num_primcaps,64), vec_order=2)

        if self.is_fc_model:
            ssr_F_Cap_model = self.ssr_FC_model_build(self.F_shape,'ssr_FC_Cap_model')
        else:
            ssr_F_Cap_model_0 = self.ssr_F_model_build(self.F_shape,'ssr_NoFC_Cap_model', vec_order=0)
            ssr_F_Cap_model_1 = self.ssr_F_model_build(self.F_shape,'ssr_NoFC_Cap_model', vec_order=1)
            ssr_F_Cap_model_2 = self.ssr_F_model_build(self.F_shape,'ssr_NoFC_Cap_model', vec_order=2)

        # Wire them up
        # ssr_G_list: [(batch_size, 8, 8, 64), (batch_size, 8, 8, 64), (batch_size, 8, 8, 64)]
        # Two-stream structure for extracting the features.

        ssr_G_list = ssr_G_model(img_inputs)

        # ssr_primcaps: (batch_size, 21, 64)
        # Generating fine-grained structure mapping from different scoring functions.
        # Apply the mapping on to the features and generate primary capsules.
        ssr_primcaps_0 = ssr_S_model_0(ssr_G_list)
        ssr_primcaps_1 = ssr_S_model_1(ssr_G_list)
        ssr_primcaps_2 = ssr_S_model_2(ssr_G_list)
        
        # ssr_Cap_list: [(None, None), (None, None), (None, None)]
        # Feed the primary capsules into capsule layer and output the final aggregated capsule features. And divide them into 3 stages.
        
        
        ssr_Cap_list_0 = ssr_aggregation_model_0(ssr_primcaps_0)
        ssr_Cap_list_1 = ssr_aggregation_model_1(ssr_primcaps_1)
        ssr_Cap_list_2 = ssr_aggregation_model_2(ssr_primcaps_2)

        print('*'*50)
        print('ssr_Cap_list_0[0]: ', ssr_Cap_list_0[0].shape)
        print('*'*50)

        # ssr_F_Cap_list: [(batch_size, 3, 3), (batch_size, 3, 3), (batch_size, 3, 3),   ~> p
        #                  (batch_size, 3),    (batch_size, 3),    (batch_size, 3),      ~> delta
        #                  (batch_size, 3),    (batch_size, 3),    (batch_size, 3)]      ~> eta
        # Taking the previous 3 stages features for Soft-Stagewise Regression (SSR) module. 
        # Each stage further splits into three parts: prediction, dynamic index shifting, and dynamic scaling. 
        # This part please check the '[IJCAI18] SSR-Net' for more detail explanation.

        # ssr_F_Cap_list_0  : (None, 3, 39)
        ssr_F_Cap_list_0 = ssr_F_Cap_model_0(ssr_Cap_list_0)
        ssr_F_Cap_list_1 = ssr_F_Cap_model_1(ssr_Cap_list_1)
        ssr_F_Cap_list_2 = ssr_F_Cap_model_2(ssr_Cap_list_2)

        print('*'*50)
        print('ssr_F_Cap_list_0', ssr_F_Cap_list_0.shape)
        print('*'*50)

        # pred_pose_l   : (None, 3)
        # Taking the prediction, dynamic index shifting, and dynamic scaling for the final regression output. In this case, there are three outputs (yaw, pitch, roll).
        pred_vec_0 = SSRLayer(s1=self.stage_num[0], s2=self.stage_num[1], s3=self.stage_num[2], lambda_d=self.lambda_d, name="pred_pose_0")(ssr_F_Cap_list_0)
        pred_vec_1 = SSRLayer(s1=self.stage_num[0], s2=self.stage_num[1], s3=self.stage_num[2], lambda_d=self.lambda_d, name="pred_pose_1")(ssr_F_Cap_list_1)
        pred_vec_2 = SSRLayer(s1=self.stage_num[0], s2=self.stage_num[1], s3=self.stage_num[2], lambda_d=self.lambda_d, name="pred_pose_2")(ssr_F_Cap_list_2)

        print('*'*50)
        print('pred_vec_0: ', pred_vec_0.shape)
        print('*'*50)

        pred_vecs = Concatenate(axis=-1)([pred_vec_0, pred_vec_1, pred_vec_2])

        print('*'*50)
        print('pred_vecs: ', pred_vecs.shape)
        print('*'*50)
        
        return Model(inputs=img_inputs, outputs=[pred_vecs, pred_vecs])
        # return Model(inputs=img_inputs, outputs=pred_pose)

# Capsule FSANetworks

class BaseCapsuleFSANet(BaseFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(BaseCapsuleFSANet, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)

    def ssr_aggregation_model_build(self, shape_primcaps, vec_order):
        input_primcaps = Input(shape_primcaps)        
        capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, routings=self.routings, name='caps')(input_primcaps)        

        feat_s1_div, feat_s2_div, feat_s3_div = AggregatedFeatureExtractionLayer(num_capsule=self.num_capsule)(capsule)

        feat_s1_div = Reshape((-1,))(feat_s1_div)
        feat_s2_div = Reshape((-1,))(feat_s2_div)
        feat_s3_div = Reshape((-1,))(feat_s3_div)        
        
        return Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model_{}'.format(vec_order))

class FSA_net_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Capsule, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)       
        self.is_varS_model = False    
 
class FSA_net_Var_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_Capsule, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)   
        self.is_varS_model = True
        
class FSA_net_noS_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_Capsule, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)   
        self.is_noS_model = True    
    
class FSA_net_Capsule_FC(FSA_net_Capsule):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Capsule_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)  
        self.is_fc_model = True

class FSA_net_Var_Capsule_FC(FSA_net_Var_Capsule):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_Capsule_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_fc_model = True
        
class FSA_net_noS_Capsule_FC(FSA_net_noS_Capsule):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_Capsule_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_fc_model = True

# Metric models

class BaseMetricFSANet(BaseFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(BaseMetricFSANet, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set) 
        
    def ssr_aggregation_model_build(self, shape_primcaps, vec_order):
        input_primcaps = Input(shape_primcaps)

        metric_feat = MatMulLayer(16,type=1)(input_primcaps)
        metric_feat = MatMulLayer(3,type=2)(metric_feat)

        feat_s1_div, feat_s2_div, feat_s3_div = AggregatedFeatureExtractionLayer(num_capsule=self.num_capsule)(metric_feat)

        feat_s1_div = Reshape((-1,))(feat_s1_div)
        feat_s2_div = Reshape((-1,))(feat_s2_div)
        feat_s3_div = Reshape((-1,))(feat_s3_div)
        
        return Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Metric_model_{}'.format(vec_order))
    
class FSA_net_Metric(BaseMetricFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Metric, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_varS_model = False    

class FSA_net_Var_Metric(BaseMetricFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_Metric, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_varS_model = True
        
class FSA_net_noS_Metric(BaseMetricFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_Metric, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_noS_model = True
   
