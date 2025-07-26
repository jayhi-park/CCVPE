import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
import math

from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

torch.manual_seed(17)
np.random.seed(0)

# ---------------------------------------------------------------------------------
# KITTI, our code is developed based on https://github.com/shiyujiao/HighlyAccurate
GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 900
GrdOriImg_W = 1600
SatMap_process_sidelength = 512

# singapore
train_file = '/ws/external/dataLoader/nuscenes_singapore/train_nuscenes.txt'
test1_file = '/ws/external/dataLoader/nuscenes_singapore/test1_nuscenes_rand.txt'
test2_file = '/ws/external/dataLoader/nuscenes_singapore/test2_nuscenes_rand.txt'

# boston
train_boston_file = '/ws/external/dataLoader/nuscenes_boston/train_nuscenes.txt'
test1_boston_file = '/ws/external/dataLoader/nuscenes_boston/test1_nuscenes_rand.txt'
test2_boston_file = '/ws/external/dataLoader/nuscenes_boston/test2_nuscenes_rand.txt'


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def get_meter_per_pixel(dataset, zoom=18, scale=1):
    if dataset in ['nuscenes', 'nuscenes_singapore']:
        lat = 1.28821008
        meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.) / (2 ** zoom)
    elif dataset in ['nuscenes_boston']:
        lat = 42.3368491
        meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.) / (2 ** zoom)
    meter_per_pixel /= 2 # because use scale 2 to get satmap
    meter_per_pixel /= scale
    return meter_per_pixel

class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10,
                 split='train'):
        dataset = 'nuscenes'
        self.root = root
        self.split = split

        self.meter_per_pixel = get_meter_per_pixel(dataset, scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        # nuScenes 데이터셋 초기화
        if split == 'test2':
            self.nusc = NuScenes(version='v1.0-test', dataroot=root, verbose=True)
        else:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=root, verbose=True)

        # 파일 리스트 로드
        # with open(file, 'r') as f:
        #     self.sample_tokens = [line.strip() for line in f.readlines()]
        with open(file, 'r') as f:
            file_name = f.readlines()
        self.file_name = [file[:-1] for file in file_name]

        sat_dir = os.path.join(self.root, 'samples/SATELLITE')
        self.sat_dict = {}
        for sat_file in os.listdir(sat_dir):
            token = sat_file.split('_')[5]
            self.sat_dict[token] = os.path.join(sat_dir, sat_file)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        line = self.file_name[idx]

        if self.split == 'train':
            sample_token = line.strip()
        else:
            sample_token, gt_shift_x, gt_shift_y, theta = line.split(' ')
        file_name = sample_token

        sample = self.nusc.get('sample', sample_token)

        # 정면 카메라 이미지 로드
        cam_front = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_front_path = os.path.join(self.root, cam_front['filename'])
        cam_image = Image.open(cam_front_path).convert('RGB')

        # 변환 적용
        if self.grdimage_transform is not None:
            cam_image = self.grdimage_transform(cam_image)

        # 카메라 intrinsic matrix 로드
        cam_calib = self.nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
        camera_k = np.array(cam_calib['camera_intrinsic'], dtype=np.float32)

        camera_k = scale_camera_intrinsic(camera_k,
                                          (GrdOriImg_H, GrdOriImg_W),
                                          (GrdImg_H, GrdImg_W))

        pointsensor = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        heading = quaternion_yaw(Quaternion(poserecord['rotation']))

        # =================== read satellite map ===================================
        SatMap_name = self.sat_dict[cam_front['ego_pose_token']]
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        sat_rot = sat_map.rotate((-heading) / np.pi * 180) # make the east direction the vehicle heading
        sat_align_cam = sat_rot

        # load the shifts
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)
        
        random_ori = float(theta) * self.rotation_range # degree
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
        
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)
        

        # gt heat map
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x, y = np.meshgrid(np.linspace(-256+x_offset,256+x_offset,512), np.linspace(-256+y_offset,256+y_offset,512))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        gt = torch.tensor(gt)
        
        # orientation gt
        orientation_angle = 90 - random_ori 
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360
        
        gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        index = int(orientation_angle // 22.5)
        ratio = (orientation_angle % 22.5) / 22.5
        if index == 0:
            gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[15, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        else:
            gt_with_ori[16-index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[16-index-1, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)
        
        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi/180))
        orientation_map[1,:,:] = np.sin(orientation_angle * np.pi/180)
        
        
        return sat_map, cam_image, gt, gt_with_ori, orientation_map, orientation_angle, file_name

def scale_camera_intrinsic(camera_k, original_resolution, new_resolution):
    """
    스케일에 맞춰 camera intrinsic matrix (camera_k)를 조정합니다.

    Parameters:
        camera_k (np.ndarray): 3x3 intrinsic matrix
        original_resolution (tuple): (width, height) of original image
        new_resolution (tuple): (width, height) of new image

    Returns:
        np.ndarray: 스케일 조정된 3x3 intrinsic matrix
    """
    orig_h, orig_w = original_resolution
    new_h, new_w = new_resolution

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    camera_k_scaled = camera_k.copy()
    camera_k_scaled[0, 0] *= scale_x  # fx
    camera_k_scaled[0, 2] *= scale_x  # cx
    camera_k_scaled[1, 1] *= scale_y  # fy
    camera_k_scaled[1, 2] *= scale_y  # cy

    return camera_k_scaled

