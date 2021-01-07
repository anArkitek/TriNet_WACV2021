import numpy as np
from scipy.optimize import minimize
import math


def isRotationMatrix(R: np.array):
    tag = False
    I = np.identity(R.shape[0])
    if np.all(np.matmul(R, R.T) - I < 1e-6) and np.linalg.det(R) - 1 < 1e-6: 
        tag = True
    return tag  


class W300():
    def W300_GetInitGuess(self, l_vec, b_vec, f_vec):
        f_vec = np.cross(b_vec, l_vec) * -1
        l_vec = np.cross(f_vec, b_vec) * -1
        
        l_norm = np.linalg.norm(l_vec)
        l_vec /= l_norm
        b_norm = np.linalg.norm(b_vec)
        b_vec /= b_norm
        f_norm = np.linalg.norm(f_vec)
        f_vec /= f_norm
        
        l_vec = l_vec.reshape(3, 1)
        b_vec = b_vec.reshape(3, 1)
        f_vec = f_vec.reshape(3, 1)
            
        l = np.array([1, 0, 0]).reshape(1, 3)
        b = np.array([0, 1, 0]).reshape(1, 3)
        f = np.array([0, 0, 1]).reshape(1, 3)
        
        R = l_vec @ l + b_vec @ b + f_vec @ f
        
        yaw = math.asin(R[0, 2])
        roll = math.atan2(-R[0, 1], R[0, 0])
        pitch = math.atan2(-R[1, 2], R[2, 2])
        yaw *= -1
        return np.array([pitch, yaw, roll])


    def W300_ObjectiveV3(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]

        l_hat, b_hat, f_hat = self.W300_EulerAngles2Vectors(rx, ry, rz)

        l_vec_dot = np.clip(l_hat[0] * l_vec[0] + l_hat[1] * l_vec[1] + l_hat[2] * l_vec[2], -1, 1)
        b_vec_dot = np.clip(b_hat[0] * b_vec[0] + b_hat[1] * b_vec[1] + b_hat[2] * b_vec[2], -1, 1) 
        f_vec_dot = np.clip(f_hat[0] * f_vec[0] + f_hat[1] * f_vec[1] + f_hat[2] * f_vec[2], -1, 1)
        
        return math.acos(l_vec_dot) ** 2 + math.acos(b_vec_dot) ** 2 + math.acos(f_vec_dot) ** 2


    def W300_EulerAngles2Vectors(self, rx, ry, rz):
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


    def W300_R2EulerAngles(self, R: np.array):
        yaw = np.arcsin(R[0, 2])
        roll = np.arctan2(-R[0, 1], R[0, 0])
        pitch = np.arctan2(-R[1, 2], R[2, 2])
        yaw *= -1
        return np.array([pitch, yaw, roll])


    def W300_Get_MAE_NPY(self, gt_mat, pred_mat, mode='svd', w2pickle=False):
        '''
        Args:
            gt_mat: (data_sample_num, 9); (left_vec, bottom_vec, front_vec)
            pred_mat: (data_sample_num, 9); (left_vec, bottom_vec, front_vec)
        '''
        c_rad2deg = 180.0 / np.pi
        c_deg2rad = np.pi / 180.0

        cnt = 0
        ccnt = 0
        roll_err_sum = 0
        pitch_err_sum = 0
        yaw_err_sum = 0
        v1_err_sum = 0
        v2_err_sum = 0
        v3_err_sum = 0

        # gt_mat = np.load(gt_inpath)
        # pred_mat = np.load(pred_inpath)

        gt_mat_deg = []
        err_mat_deg = []

        
        for row in range(len(gt_mat)):
            gt_v1 = gt_mat[row, : 3]
            gt_v2 = gt_mat[row, 3: 6]
            gt_v3 = gt_mat[row, 6: 9]
            pred_v1 = pred_mat[row, : 3]
            pred_v2 = pred_mat[row, 3: 6]
            pred_v3 = pred_mat[row, 6: 9]

            # print(np.linalg.norm(gt_v1))
            # print(np.linalg.norm(gt_v2))
            # print(np.linalg.norm(gt_v3))
            assert np.linalg.norm(gt_v1) - 1.0 < 1e-6
            assert np.linalg.norm(gt_v2) - 1.0 < 1e-6
            assert np.linalg.norm(gt_v3) - 1.0 < 1e-6

            pred_v1 /= np.linalg.norm(pred_v1)
            pred_v2 /= np.linalg.norm(pred_v2)
            pred_v3 /= np.linalg.norm(pred_v3)
            
            # GROUND TRUTH
            R_gt = np.array([[gt_v1[0], gt_v2[0], gt_v3[0]],
                             [gt_v1[1], gt_v2[1], gt_v3[1]],
                             [gt_v1[2], gt_v2[2], gt_v3[2]]])
            p_gt_deg, y_gt_deg, r_gt_deg = self.W300_R2EulerAngles(R_gt) * c_rad2deg


            # PREDICTION
            p_pred_deg, y_pred_deg, r_pred_deg = None ,None, None
            # --------------------------------- OPTIMIZE ---------------------------------#
            # if mode == 'optimize':
            #     x0 = self.W300_GetInitGuess(pred_v1, pred_v2, pred_v3)
            #     sol = minimize(self.W300_ObjectiveV3, x0, args=(pred_v1, pred_v2, pred_v3), method='nelder-mead', options={'xatol': 1e-7, 'disp': False})
            #     p_pred_deg, y_pred_deg, r_pred_deg = sol.x * c_rad2deg

            # --------------------------------- SVD ---------------------------------#
            if mode == 'svd':
                R_pred = np.array([[pred_v1[0], pred_v2[0], pred_v3[0]],
                                [pred_v1[1], pred_v2[1], pred_v3[1]],
                                [pred_v1[2], pred_v2[2], pred_v3[2]]])
                U, Sig, V_T = np.linalg.svd(R_pred)
                R_hat = U @ V_T
                assert isRotationMatrix(R_hat)
                p_pred_deg, y_pred_deg, r_pred_deg = self.W300_R2EulerAngles(R_hat) * c_rad2deg

            # Vector errors
            # print(np.sum(gt_v2 * pred_v2))
            v1_err = math.acos(np.clip(np.sum(gt_v1 * pred_v1), -1, 1)) * c_rad2deg
            v2_err = math.acos(np.clip(np.sum(gt_v2 * pred_v2), -1, 1)) * c_rad2deg
            v3_err = math.acos(np.clip(np.sum(gt_v3 * pred_v3), -1, 1)) * c_rad2deg

            # Euler angle errors
            pitch_err_deg = min( abs(p_gt_deg - p_pred_deg), abs(p_pred_deg + 360 - p_gt_deg), abs(p_pred_deg - 360 - p_gt_deg), abs(p_pred_deg + 180 - p_gt_deg), abs(p_pred_deg - 180 - p_gt_deg))
            yaw_err_deg = min( abs(y_gt_deg - y_pred_deg), abs(y_pred_deg + 360 - y_gt_deg), abs(y_pred_deg - 360 - y_gt_deg), abs(y_pred_deg + 180 - y_gt_deg), abs(y_pred_deg - 180 - y_gt_deg))
            roll_err_deg = min( abs(r_gt_deg - r_pred_deg), abs(r_pred_deg + 360 - r_gt_deg), abs(r_pred_deg - 360 - r_gt_deg), abs(r_pred_deg + 180 - r_gt_deg), abs(r_pred_deg - 180 - r_gt_deg))
            
            
            gt_mat_deg.append([p_gt_deg, y_gt_deg, r_gt_deg])
            err_mat_deg.append([pitch_err_deg, yaw_err_deg, roll_err_deg])
        

            # Accumulation
            pitch_err_sum += pitch_err_deg
            yaw_err_sum += yaw_err_deg
            roll_err_sum += roll_err_deg
            v1_err_sum += v1_err
            v2_err_sum += v2_err
            v3_err_sum += v3_err
            cnt += 1

        
        print('VECTORS ERROR:')
        print(v1_err_sum / cnt)
        print(v2_err_sum / cnt)
        print(v3_err_sum / cnt)
        MAEV = (v1_err_sum + v2_err_sum + v3_err_sum) / (3 * cnt)
        print(MAEV)
        print('-'*10)
        print('EULER ANGLES ERROR')
        print(roll_err_sum / cnt)
        print(pitch_err_sum / cnt)
        print(yaw_err_sum / cnt)
        print((roll_err_sum + pitch_err_sum + yaw_err_sum) / (3 * cnt))
        
        print('='*10)
        print('cnt:', cnt)

        #--------------------------- Save to the pickle --------------------------------#
        if w2pickle:
            with open('./pickles/ground_truth.pickle', 'wb') as handle:
                pickle.dump(gt_mat_deg, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./pickles/prediction.pickle', 'wb') as handle:
                pickle.dump(err_mat_deg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------------------------------------------------#

        return gt_mat_deg, err_mat_deg, MAEV




    def W300_Get_MAE(self, pred_inpath, gt_inpath, ):
        
        err_dict = {'roll': list(), 'pitch': list(), 'yaw': list()}
        roll_err_dict = {'ground_truth': list(), 'error': list()}
        pitch_err_dict = {'ground_truth': list(), 'error': list()}
        yaw_err_dict = {'ground_truth': list(), 'error': list()}

        c_rad2deg = 180.0 / np.pi
        c_deg2rad = np.pi / 180.0

        files = sorted([f for f in os.listdir(pred_inpath) if os.path.isfile(os.path.join(pred_inpath, f)) and f.endswith('.txt')])

        cnt = 0
        ccnt = 0
        roll_err_sum = 0
        pitch_err_sum = 0
        yaw_err_sum = 0
        l_vec_err_sum = 0
        b_vec_err_sum = 0
        f_vec_err_sum = 0
        
        x_min, y_min, x_max, y_max = None, None, None, None

        for fname in files:
            roll_deg, pitch_deg, yaw_deg = None, None, None
            pitch_err, yaw_err, roll_err = None, None, None
            with open(os.path.join(pred_inpath, fname), 'r') as f:
                line = f.readline()
                l0, l1, l2 = np.array(list(map(float, line.split(' '))))
                line = f.readline()
                b0, b1, b2 = np.array(list(map(float, line.split(' '))))
                line = f.readline()
                f0, f1, f2 = np.array(list(map(float, line.split(' '))))

                l_vec_pred = np.array([float(l0), float(l1), float(l2)])
                b_vec_pred = np.array([float(b0), float(b1), float(b2)])
                f_vec_pred = np.array([float(f0), float(f1), float(f2)])

                # --------------------------------- Optimize ---------------------------------#
                x0 = self.W300_GetInitGuess(l_vec_pred, b_vec_pred, f_vec_pred)
                sol = minimize(self.W300_ObjectiveV3, x0, args=(l_vec_pred, b_vec_pred, f_vec_pred), method='nelder-mead', options={'xatol': 1e-7, 'disp': False})
                p_pred_deg, y_pred_deg, r_pred_deg = sol.x * c_rad2deg

                # --------------------------------- SVD ---------------------------------#
                # R = np.array([ [l0, b0, f0],
                #                [l1, b1, f1],
                #                [l2, b2, f2] ], dtype='float')
                # U, Sig, V_T = np.linalg.svd(R)
                # R_hat = U @ V_T
                # assert isRotationMatrix(R_hat)
                # p_pred_deg, y_pred_deg, r_pred_deg = w300.W300_R2EulerAngles(R_hat) * c_rad2deg


                # --------------------------------- Two Vectors ---------------------------------#

                


                # ------------------------------ Chart ------------------------------#
                # if fname == 'image00035_0.txt':
                #     print('Predicted vectors:')
                #     print(str(l_vec))
                #     print(str(b_vec))
                #     print(str(f_vec))

                #     l_vec_new, b_vec_new, f_vec_new = W300_EulerAngles2Vectors(*sol.x)
                #     print('After Optimization:')
                #     print(str(l_vec_new))
                #     print(str(b_vec_new))
                #     print(str(f_vec_new))

            with open(os.path.join(gt_inpath, fname), 'r') as f:
                line = f.readline()
                p_gt_deg, y_gt_deg, r_gt_deg = list(map(float, line.split(',')))
                line = f.readline()
                
                line = f.readline()
                l_vec_gt = np.array(list(map(float, line.split(','))))
                line = f.readline()
                b_vec_gt = np.array(list(map(float, line.split(','))))
                line = f.readline()
                f_vec_gt = np.array(list(map(float, line.split(','))))

                R_gt = np.stack([l_vec_gt, b_vec_gt, f_vec_gt]).T
                assert isRotationMatrix(R_gt)

                #r_gt_deg, p_gt_deg, y_gt_deg = w300.W300_R2EulerAngles(R_gt) * c_rad2deg
                p_gt_deg_from_R, y_gt_deg_from_R, r_gt_deg_from_R = self.W300_R2EulerAngles(R_gt) * c_rad2deg
                
                # discard numerically unstable ones
                if abs(p_gt_deg - p_gt_deg_from_R) > 1 or (y_gt_deg - y_gt_deg_from_R) > 1 or (r_gt_deg - r_gt_deg_from_R) > 1:
                    # print('pitch: ', p_gt_deg - p_gt_deg_from_R,
                    #       'yaw: ', y_gt_deg - y_gt_deg_from_R,
                    #       'roll: ', r_gt_deg - r_gt_deg_from_R)
                    continue
                
                # vector errors in degrees
                l_vec_err = math.acos(np.sum(l_vec_gt * l_vec_pred)) * c_rad2deg
                b_vec_err = math.acos(np.sum(b_vec_gt * b_vec_pred)) * c_rad2deg
                f_vec_err = math.acos(np.sum(f_vec_gt * f_vec_pred)) * c_rad2deg


                pitch_err_deg = abs(p_gt_deg - p_pred_deg)
                yaw_err_deg = abs(y_gt_deg - y_pred_deg)
                roll_err_deg = abs(r_gt_deg - r_pred_deg)

                # ------------------------------ Consider large angles ------------------------------#
                # if p_gt_deg > 99 or p_gt_deg < -99:
                #     continue
                # if y_gt_deg > 90 or y_gt_deg < -90:
                #     continue
                # if r_gt_deg > 99 or r_gt_deg < -99:
                #     continue

                # ------------------------------ Select pictures with large errors? ------------------------------#
                # if pitch_err_deg > 10 or yaw_err_deg > 10 or roll_err_deg > 10:
                #     # print('pitch: ',p_gt_deg, ' yaw: ', y_gt_deg, ' roll: ', r_gt_deg)
                #     img = cv2.imread(os.path.join('AFLW2000(300W)_GT/imgs/', fname.split('.')[0] + '.jpg'))
                #     draw_vector(img, R_hat[0, 0], R_hat[1, 0], color='r')
                #     draw_vector(img, R_hat[0, 1], R_hat[1, 1], color='g')
                #     draw_vector(img, R_hat[0, 2], R_hat[1, 2], color='b')
                #     cv2.imwrite(os.path.join('AFLW2000(300W)_GT/temp/', fname.split('.')[0] + '.jpg'), img)
                #     ccnt += 1
                    
                cnt += 1

                #--------------------------- Save to the pickle --------------------------------#
                # err_dict['roll'].append(roll_err)
                # err_dict['pitch'].append(pitch_err)
                # err_dict['yaw'].append(yaw_err)
                roll_err_dict['ground_truth'].append(r_gt_deg)
                pitch_err_dict['ground_truth'].append(p_gt_deg)
                yaw_err_dict['ground_truth'].append(y_gt_deg)

                roll_err_dict['error'].append(roll_err_deg)
                pitch_err_dict['error'].append(pitch_err_deg)
                yaw_err_dict['error'].append(yaw_err_deg)
                #--------------------------------------------------------------------------------#

                # compute accumulated errors
                pitch_err_sum += pitch_err_deg
                yaw_err_sum += yaw_err_deg
                roll_err_sum += roll_err_deg
                l_vec_err_sum += l_vec_err
                b_vec_err_sum += b_vec_err
                f_vec_err_sum += f_vec_err

        
        #--------------------------- Save to the pickle --------------------------------#
        with open('roll_error_diagram.pickle', 'wb') as handle:
            pickle.dump(roll_err_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('pitch_error_diagram.pickle', 'wb') as handle:
            pickle.dump(pitch_err_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('yaw_error_diagram.pickle', 'wb') as handle:
            pickle.dump(yaw_err_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------------------------------------------------#

        print('VECTORS ERROR:')
        print(l_vec_err_sum / cnt)
        print(b_vec_err_sum / cnt)
        print(f_vec_err_sum / cnt)
        print((l_vec_err_sum + b_vec_err_sum + f_vec_err_sum) / (3 * cnt))
        print('-'*10)
        print('EULER ANGLES ERROR')
        print(roll_err_sum / cnt)
        print(pitch_err_sum / cnt)
        print(yaw_err_sum / cnt)
        print((roll_err_sum + pitch_err_sum + yaw_err_sum) / (3 * cnt))
        
        print('='*10)
        print('cnt:', cnt)
        print('ccnt:', ccnt)

