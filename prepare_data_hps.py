'''
# --------------------------------------------
# data preprocessing for AMASS dataset
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''
import torch
import numpy as np
import os
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose
from utils import utils_transform
import time
import pickle
import json
import cv2

def linear_interpolate(data, index, length):
    if index == 0:
        data[index] = data[index+1]
    elif index == length-1:
        data[index] = data[index-1]
    else:
        data[index] = (data[index-1] + data[index+1]) / 2
    return data

dataset_hps ="/local/home/liuqing/SP/dataset_hps" # root of amass dataset

path_head_camera_localizations = dataset_hps + "/head_camera_localizations"

# read the camera file.
camera_path = dataset_hps + '/head_camera_videos/cameras.json'

try:
    with open(camera_path, 'r') as file:
        camera_data = json.load(file)
except:
    print("Camera file not found")

beta_path = dataset_hps + '/hps_betas/'   
betas = {}
for beta_file in os.listdir(beta_path):
    if beta_file.endswith('.json'):
        with open(os.path.join(beta_path, beta_file), 'r') as file:
            beta_data = json.load(file)
            betas[beta_file[3]] = beta_data
            



camera_ids = ['029756', '029757']

K_dict = {}

for camera_id in camera_ids:
    focal_dist_xy = camera_data[camera_id]['parameters']['focal_dist_xy']
    center_xy = camera_data[camera_id]['parameters']['center_xy']
    K_dict[camera_id] = np.array([[focal_dist_xy[0], 0, center_xy[0]], [0, focal_dist_xy[1], center_xy[1]], [0, 0, 1]])

for phase in ["train","test"]:
    print(phase)
    savedir = os.path.join("./data_fps60_hps", phase)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not os.path.exists('./data_split_hps'):
        os.makedirs('./data_split_hps')
    

    split_file = os.path.join("./data_split_hps", phase+"_split.txt")

    with open(split_file, 'r') as f:
        filenames = [line.rstrip('\n') for line in f]

    rotation_local_full_gt_list = []

    hmd_position_global_full_gt_list = []

    body_parms_list = []

    head_global_trans_list = []

    support_dir = 'support_data/'
    bm_fname_male = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('male'))
    dmpl_fname_male = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('male'))

    bm_fname_female = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('female'))
    dmpl_fname_female = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('female'))

    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    bm_male = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male)#.to(comp_device)
    bm_female = BodyModel(bm_fname=bm_fname_female, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_female)

    bm = bm_male if beta_data['gender'] == 'male' else bm_female
    
    idx = 0
    
    for filename in filenames:
        print(filename)
        data = dict()
        for camera_id in camera_ids:
            video_names = [x[:-4] for x in camera_data[camera_id]['captured_videos']]
            if filename in video_names:
                camera = camera_id
        
        K = K_dict[camera]
        dist_coeffs = camera_data[camera_id]['parameters']['dist_coeffs']
        
        inits_path = dataset_hps + '/hps_inits/' + filename + '.json'   
        with open(inits_path, 'r') as file:
            inits_data = json.load(file)
        
        cam_start = inits_data['cam_start']
        imu_start = inits_data['imu_start']

        head_pose_path = dataset_hps + '/head_camera_localizations/' + filename + '.json'
        with open(head_pose_path, 'r') as file:
            head_pose_data = json.load(file)

        body_pose_path = dataset_hps + '/hps_txt/' + filename + '_pose.txt'
        body_trans_path = dataset_hps + '/hps_txt/' + filename + '_trans.txt'

        body_pose_data = np.loadtxt(body_pose_path)[imu_start:] 
        body_trans_data = np.loadtxt(body_trans_path)[imu_start:] 

        # if filename starts with "D"
        if filename[0] == 'D': # for files starting with "Double_"
            beta_data = betas[filename[10]]
        else:
            beta_data = betas[filename[3]]


        idx+=1

        video_path = dataset_hps + '/head_camera_videos/' + filename + '.mp4'   
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        # print(fps) # 29.97002997002997
        # convert video to frames
        # success, image = video.read()
        # data['images'] = image
        # print(image.shape) # (1080, 1920, 3)
        # 初始化一个列表来保存所有帧
        frames = []

        frame_count = 0
        # read total frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total video frames: {total_frames}")
        while True:
            success, image = video.read()
            if not success:
                break  
            frames.append(image)  
            frame_count += 1
            if frame_count % 1000 == 0:  
                print(frame_count, '/', total_frames)
            
            # print(image.shape)  
        print(len(frames)) 
        video.release() 


        ########################################################################################
        
        # body_parms = {
        #         'root_orient': torch.Tensor(body_pose_data[:, :3]),#.to(comp_device), # controls the global root orientation
        #         'pose_body': torch.Tensor(body_pose_data[:, 3:66]),#.to(comp_device), # controls the body
        #         'trans': torch.Tensor(body_trans_data),#.to(comp_device), # controls the global body position
        #     }

        # # print(body_parms['root_orient'].shape, body_parms['pose_body'].shape, body_parms['trans'].shape)

        # body_parms_list = body_parms
        # body_pose_world=bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient','trans']})


        # output_aa = torch.Tensor(body_pose_data[:, :66]).reshape(-1,3)
        # output_6d = utils_transform.aa2sixd(output_aa).reshape(body_pose_data.shape[0],-1)
        # rotation_local_full_gt_list = output_6d[1:]
        # # print(bm.kintree_table[0])  # tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        # #     16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        # #     35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50],
        # #    dtype=torch.int32)
        # rotation_local_matrot = aa2matrot(torch.tensor(body_pose_data).reshape(-1, 3)).reshape(body_pose_data.shape[0],-1,9)

        # filtered_kintree_table = bm.kintree_table[0][:25] # only consider the first 25 joints

        # rotation_global_matrot = local2global_pose(rotation_local_matrot, filtered_kintree_table.long()) # rotation of joints relative to the origin

        # head_rotation_global_matrot = rotation_global_matrot[:,[15],:,:]
    

        # rotation_global_6d = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1,3,3)).reshape(rotation_global_matrot.shape[0],-1,6)
        # # print(rotation_global_6d.shape) # torch.Size([3154, 52, 6])
        # input_rotation_global_6d = rotation_global_6d[1:,[15,20,21],:]
        # # print(input_rotation_global_6d.shape) # torch.Size([3153, 3, 6])

        # rotation_velocity_global_matrot = torch.matmul(torch.inverse(rotation_global_matrot[:-1]),rotation_global_matrot[1:])
        # rotation_velocity_global_6d = utils_transform.matrot2sixd(rotation_velocity_global_matrot.reshape(-1,3,3)).reshape(rotation_velocity_global_matrot.shape[0],-1,6)
        # input_rotation_velocity_global_6d = rotation_velocity_global_6d[:,[15,20,21],:]

        # position_global_full_gt_world = body_pose_world.Jtr[:,:22,:] # position of joints relative to the world origin

        # position_head_world = position_global_full_gt_world[:,15,:] # world position of head

        # # print(position_head_world.shape) # torch.Size([3154, 3])
        
        # head_global_trans = torch.eye(4).repeat(position_head_world.shape[0],1,1)
        # head_global_trans[:,:3,:3] = head_rotation_global_matrot.squeeze()
        # head_global_trans[:,:3,3] = position_global_full_gt_world[:,15,:]

        # head_global_trans_list = head_global_trans[1:]


        # num_frames = position_global_full_gt_world.shape[0]-1

        # # print(input_rotation_global_6d.reshape(num_frames,-1).shape)
        # hmd_position_global_full_gt_list = torch.cat([
        #                                                         input_rotation_global_6d.reshape(num_frames,-1),
        #                                                         input_rotation_velocity_global_6d.reshape(num_frames,-1),
        #                                                         position_global_full_gt_world[1:, [15,20,21], :].reshape(num_frames,-1), 
        #                                                         position_global_full_gt_world[1:, [15,20,21], :].reshape(num_frames,-1)-position_global_full_gt_world[:-1, [15,20,21], :].reshape(num_frames,-1)], dim=-1)
        # # print(hmd_position_global_full_gt_list.shape) # torch.Size([3153, 54])
        # data_count = len(hmd_position_global_full_gt_list)

        


        # data['rotation_local_full_gt_list'] = rotation_local_full_gt_list

        # data['hmd_position_global_full_gt_list'] = hmd_position_global_full_gt_list

        # data['body_parms_list'] = body_parms_list

        # data['head_global_trans_list'] = head_global_trans_list

        # data['framerate'] = 30 # TODO: check the framerate

        # data['gender'] = beta_data['gender']


        ########################################################################################



        # camera_trans_list = []
        # camera_6d_list = []
        # total_frames = len(head_pose_data)
        # # print('total_frames', total_frames)
        # for frame_id, frame in head_pose_data.items():
        #     if frame is None:

        #         if int(frame_id) == total_frames-1:

        #             frame = {}
        #             frame['position'] = head_pose_data[str(int(frame_id) - 1)]['position']
        #             frame['quaternion'] = head_pose_data[str(int(frame_id) - 1)]['quaternion']
        #         else:

        #             for next_id in range(int(frame_id) + 1, total_frames):

        #                 # print('next_id', next_id)
        #                 if head_pose_data[str(next_id)] is not None:
                            
        #                     next_frame = head_pose_data[str(next_id)]
        #                     # print(f"Found a non-None frame for {frame_id} at frame_id {next_id}")
        #                     break
        #                 elif next_id == total_frames:
        #                     next_id = total_frames
        #                     print(f"Couldn't find a non-None frame for {frame_id}")
        #                     break
                    
        #             frame = {}
        #             current_id = int(frame_id)
    
        #             start_position = head_pose_data[str(current_id-1)]['position']
        #             end_position = head_pose_data[str(next_id)]['position']
        #             start_quaternion = head_pose_data[str(current_id-1)]['quaternion']
        #             end_quaternion = head_pose_data[str(next_id)]['quaternion']
        #             # Number of missing frames to interpolate
        #             num_missing_frames = next_id - current_id
        #             # Linear interpolation for positions
        #             interpolated_positions = [
        #                 np.linspace(start_position[i], end_position[i], num_missing_frames + 2)
        #                 for i in range(3)
        #             ]
        #             # Linear interpolation for quaternions
        #             interpolated_quaternions = [
        #                 np.linspace(start_quaternion[i], end_quaternion[i], num_missing_frames + 2)
        #                 for i in range(4)
        #             ]
        #             # print(num_missing_frames)
        #             # Insert interpolated values into head_pose_data for the missing frames
        #             for i in range(1, num_missing_frames + 1):
        #                 interpolated_frame_id = str(current_id + i - 1)

        #                 head_pose_data[interpolated_frame_id] = {
        #                     'position': [interpolated_positions[j][i] for j in range(3)],
        #                     'quaternion': [interpolated_quaternions[j][i] for j in range(4)]
        #                 }


        # for frame_id, frame in head_pose_data.items():
        #     # print('processing frame:', frame_id)
        #     if int(frame_id) <= cam_start-1:
        #         pass
        #     else:
        #         if frame is None:
        #             continue
        #         else:
        #             camera_trans_list.append(frame['position'])
        #             quat = frame['quaternion']
        #             rot = utils_transform.quat2aa(torch.tensor(quat).reshape(1,4)).numpy()
        #             sixd = utils_transform.aa2sixd(torch.tensor(rot).reshape(1,3)).numpy()
        #             # print(sixd.shape)
        #             camera_6d_list.append(sixd)
        

        # print(f"sixd: {len(camera_6d_list)}, trans', {len(camera_trans_list)}, total_frames: {total_frames}")


        #########################################


        print(str(idx)+'/'+str(len(filenames)))