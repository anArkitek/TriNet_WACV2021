import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('..')
import logging
import argparse

from computeMAE import *

import numpy as np
import pandas as pd
import pickle

from keras import backend as K
from keras.layers import *
from keras.utils import plot_model
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from lib.FSANET_model import *
from lib.SSRNET_model import *

import matplotlib.pyplot as plt

import TYY_callbacks
from TYY_generators import *

_TRAIN_DB_300W_LP = "300W_LP"
_TRAIN_DB_BIWI = "BIWI"

_TEST_DB_AFLW = "AFLW2000"
_TEST_DB_BIWI = "BIWI"

_IMAGE_SIZE = 64

def load_data_npz(npz_path):
    def W300_EulerAngles2Vectors(rx, ry, rz):
        '''
        rx: pitch
        ry: yaw
        rz: roll
        '''
        ry *= -1
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx)],
                        [0.0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                        [np.sin(rz), np.cos(rz), 0.0],
                        [0.0, 0.0, 1.0]])

        R = R_x @ R_y @ R_z
        
        l_vec = R @ np.array([1, 0, 0]).T
        b_vec = R @ np.array([0, 1, 0]).T
        f_vec = R @ np.array([0, 0, 1]).T
        return l_vec, b_vec, f_vec


    d = np.load(npz_path)
    poses = d['pose']
    l_vecs = []
    b_vecs = []
    f_vecs = []
    vecs = []

    for pose in poses:
        yaw_rad, pitch_rad, roll_rad = pose * np.pi / 180.0
        temp_l_vec, temp_b_vec, temp_f_vec = W300_EulerAngles2Vectors(pitch_rad, yaw_rad, roll_rad)
        vecs.append([temp_l_vec[0], temp_l_vec[1], temp_l_vec[2], 
                     temp_b_vec[0], temp_b_vec[1], temp_b_vec[2],
                     temp_f_vec[0], temp_f_vec[1], temp_f_vec[2]])

    # print('='*50)
    # print('vecs: ', np.array(vecs).shape)
    # print('='*50)

    return d["image"], np.array(vecs), poses


def vector_angle(m1,m2):
    angle_errors = np.degrees(np.arccos(np.sum(m1*m2, axis=1)/ (np.linalg.norm(m1, axis=1)*np.linalg.norm(m2, axis=1))))
    return np.mean(angle_errors), angle_errors


def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass

def get_weights_file_path(use_pretrained, train_db_name, save_name):
    prefix = "../pre-trained/" if use_pretrained else ""
    return prefix + train_db_name+"_models/"+save_name+"/"+save_name+".h5"    

def get_args():
    parser = argparse.ArgumentParser(description="This script tests the CNN model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_type", type=int, default=3, 
                        help="type of model")
    parser.add_argument('--use_pretrained', required=False,
                        dest='use_pretrained',
                        action='store_true')
    parser.add_argument("--train_db", choices=[_TRAIN_DB_300W_LP, _TRAIN_DB_BIWI], required=False, default=_TRAIN_DB_BIWI)

    parser.set_defaults(use_pretrained=False)

    args = parser.parse_args()
    return args

def main():
    K.clear_session()
    K.set_learning_phase(0) # make sure its testing mode
    
    args = get_args()
    
    model_type = args.model_type
    train_db_name = args.train_db
    use_pretrained = args.use_pretrained   
    

    if train_db_name == _TRAIN_DB_300W_LP:
        test_db_list = [_TEST_DB_AFLW]
    elif train_db_name == _TRAIN_DB_BIWI:
        test_db_list = [_TEST_DB_BIWI]

    for test_db_name in test_db_list:

        if test_db_name == _TEST_DB_AFLW:            
            image, pose, Euler_angles = load_data_npz('../data/type1/AFLW2000.npz')
        elif test_db_name == _TEST_DB_BIWI:            
            if train_db_name == _TRAIN_DB_300W_LP:
                image, pose, Euler_angles = load_data_npz('../data/BIWI_noTrack.npz')
            elif train_db_name == _TRAIN_DB_BIWI:
                image, pose, Euler_angles = load_data_npz('../data/BIWI_test.npz')
        
        if train_db_name == _TRAIN_DB_300W_LP:
            # we only care the angle between [-99,99] and filter other angles
            x_data = []
            y_data = []
            angles = []

            for i in range(0,pose.shape[0]):
                temp_pose = pose[i,:]
                #temp_pose = Euler_angles[i,:]
                #if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
                if temp_pose[0] >= 0 and temp_pose[8] >= 0:
                #if -180.0 < temp_pose[1] < 180.0 and -90.0 <  temp_pose[0] < 90.0 and -180.0 < temp_pose[2] < 180.0:
                    x_data.append(image[i,:,:,:])
                    y_data.append(pose[i,:])
                    angles.append(Euler_angles[i,:])
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            angles = np.array(angles)
        else:
            x_data = image
            y_data = pose
        
        stage_num = [3,3,3]
        lambda_d = 1
        num_classes = 3

        if model_type == 0:
            model = SSR_net_ori_MT(_IMAGE_SIZE, num_classes, stage_num, lambda_d)()
            save_name = 'ssrnet_ori_mt'

        elif model_type == 1:
            model = SSR_net_MT(_IMAGE_SIZE, num_classes, stage_num, lambda_d)()
            save_name = 'ssrnet_mt'

        elif model_type == 2:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_capsule'+str_S_set
        
        elif model_type == 3:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_capsule'+str_S_set
        elif model_type == 4:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_netvlad'+str_S_set
        elif model_type == 5:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_netvlad'+str_S_set
        elif model_type == 6:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_capsule'+str_S_set
        elif model_type == 7:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_netvlad'+str_S_set
        
        elif model_type == 8:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_capsule_fine'+str_S_set
            
        elif model_type == 9:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_capsule_fc'+str_S_set
        elif model_type == 10:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_capsule_fc'+str_S_set
        elif model_type == 11:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_capsule_fc'+str_S_set
        elif model_type == 12:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_netvlad_fc'+str_S_set
        elif model_type == 13:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_netvlad_fc'+str_S_set
        elif model_type == 14:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_netvlad_fc'+str_S_set

        elif model_type == 15:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_capsule'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_capsule'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_capsule'+str_S_set
            save_name = 'fusion_dim_split_capsule'
        elif model_type == 16:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_capsule_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_capsule_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_capsule_fc'+str_S_set

            save_name = 'fusion_fc_capsule'
        elif model_type == 17:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_netvlad'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_netvlad'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_netvlad'+str_S_set

            save_name = 'fusion_dim_split_netvlad'
        elif model_type == 18:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_netvlad_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_netvlad_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_netvlad_fc'+str_S_set
            save_name = 'fusion_fc_netvlad'

        elif model_type == 19:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = "".join('_'+str(x) for x in S_set)

            model = FSA_net_noS_Metric(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsa_noS_metric'+str_S_set

        elif model_type == 20:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_Metric(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()

        if model_type == 20:
            model.load_weights('./BIWI_models/S_noCap_noOrtho/weights.90-0.03.hdf5')

        elif model_type <15:            
            weight_file = get_weights_file_path(use_pretrained, train_db_name, save_name)
            model.load_weights(weight_file)
        else:            
            weight_file1 = get_weights_file_path(use_pretrained, train_db_name, save_name1)
            model1.load_weights(weight_file1)            
            weight_file2 = get_weights_file_path(use_pretrained, train_db_name, save_name2)
            model2.load_weights(weight_file2)            
            weight_file3 = get_weights_file_path(use_pretrained, train_db_name, save_name3)
            model3.load_weights(weight_file3)
            inputs = Input(shape=(64,64,3))
            x1 = model1(inputs)
            x2 = model2(inputs)
            x3 = model3(inputs)
            outputs = Average()([x1,x2,x3])
            model = Model(inputs=inputs,outputs=outputs)

        p_data,_ = model.predict(x_data)
        print(y_data.shape)
        #print(p_data[:5])
        #print(y_data[:5])

        #Euler_angles[:,[0,1]] = Euler_angles[:,[1,0]]
        #save = np.concatenate([p_data, Euler_angles], axis=1)
        #print(save.shape)
        #with open('BIWI_trinet.npy','wb') as f:
        #    np.save(f, save)


        #compute vector errors
        #l_MAE, l_errors = vector_angle(p_data[:,:3], y_data[:,:3])

        #b_MAE, b_errors = vector_angle(p_data[:,3:6], y_data[:,3:6])

        #f_MAE, f_errors = vector_angle(p_data[:,6:9], y_data[:,6:9])

        #computer euler angle(pitch, yaw, roll)
        w300 = W300()
        #save = {}
        #for i in range(p_data.shape[0]):
        #    tmp = []
        #    tmp.extend(list(p_data[i,:]))
        #    tmp.extend(list(y_data[i,:]))
        #    save[name[i]] = tmp

        #with open("trinet.pickle", 'wb') as f:
        #    pickle.dump(save, f)
        gt_angles, angle_errorsi, _ = w300.W300_Get_MAE_NPY(y_data, p_data)
        #gt_angles = np.array(gt_angles)
        #angle_errors = np.array(angle_errors)
        #print(gt_angles.shape)
        #print(angle_errors.shape)

        #np.save('AFLW2000.npy', y_data)
        #angles[:,[0,1]] = angles[:,[1,0]]
        #save = np.concatenate([p_data,angles],axis=1)

        #np.save('Prediction.npy', save)
        #print(save.shape)

        #visualize vector errors
        """
        print("Start plotting errors...")
        f, ax = plt.subplots(2,3, figsize=(30,20))
        ax[0,0].scatter(y_data[:,0], l_errors.reshape(l_errors.shape[0],-1))
        ax[0,0].set_ylabel("left vector error")
        ax[0,0].set_ylim([0,80])

        ax[0,1].scatter(y_data[:,3], b_errors.reshape(b_errors.shape[0],-1))
        ax[0,1].set_ylabel("down vector error")
        ax[0,1].set_ylim([0,80])

        ax[0,2].scatter(y_data[:,8], f_errors.reshape(f_errors.shape[0],-1))
        ax[0,2].set_ylabel("front vector error")
        ax[0,2].set_ylim([0,80])


        print("left vector error:{}".format(l_MAE))
        print("bottom vector error:{}".format(b_MAE))
        print("front vector error:{}".format(f_MAE))

        #visualize euler angle errors
        ax[1,0].scatter(gt_angles[:,0], angle_errors[:,0])
        ax[1,0].set_ylabel("pitch errors")
        ax[1,0].set_xlabel("gt pitch")
        ax[1,0].set_ylim([0,80])


        ax[1,1].scatter(gt_angles[:,1], angle_errors[:,1])
        ax[1,1].set_ylabel("yaw errors")
        ax[1,1].set_xlabel("gt pitch")
        ax[1,1].set_ylim([0,80])
        
        ax[1,2].scatter(gt_angles[:,2], angle_errors[:,2])
        ax[1,2].set_ylabel("roll errors")
        ax[1,2].set_xlabel("gt roll")
        ax[1,2].set_ylim([0,80])

        print("pitch MAE:", np.mean(angle_errors[:,0]))
        print("yaw MAE:", np.mean(angle_errors[:,1]))
        print("roll MAE:", np.mean(angle_errors[:,2]))

        print("Finished plotting errors, saving...")
        plt.savefig('errorSummary.png')
        print("Saved!")
        """



if __name__ == '__main__':    
    main()
