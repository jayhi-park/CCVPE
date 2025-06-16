import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 
os.environ["OMP_NUM_THREADS"] = "4" 

import argparse
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from datasets import SatGrdDatasetDemo
from losses import infoNCELoss, cross_entropy_loss, orientation_loss
from models import CVM_KITTI as CVM

import time

torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--weight_ori', type=float, help='weight on orientation loss', default=1e1)
parser.add_argument('--weight_infoNCE', type=float, help='weight on infoNCE loss', default=1e4)
parser.add_argument('--shift_range_lat', type=float, help='range for random shift in lateral direction', default=20)
parser.add_argument('--shift_range_lon', type=float, help='range for random shift in longitudinal direction', default=20)
parser.add_argument('--rotation_range', type=float, help='range for random orientation', default=180)
parser.add_argument('--data_date', type=str, default='2011_09_29', help='date for test')

args = vars(parser.parse_args())
learning_rate = args['learning_rate']
batch_size = args['batch_size']
weight_ori = args['weight_ori']
weight_infoNCE = args['weight_infoNCE']
shift_range_lat = args['shift_range_lat']
shift_range_lon = args['shift_range_lon']
rotation_range = args['rotation_range']
data_date = args['data_date']

label = 'KITTI_rotation_range' + str(rotation_range)

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 1

dataset_root = '/ws/data/kitti-vo'


SatMap_original_sidelength = 512 
SatMap_process_sidelength = 512 


satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
grdimage_transform = transforms.Compose([
        transforms.Resize(size=[GrdImg_H, GrdImg_W]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

demo_set = SatGrdDatasetDemo(root=dataset_root, data_date=data_date,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

demo_loader = DataLoader(demo_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)



torch.cuda.empty_cache()
CVM_model = CVM(device)

# test
test_model_path = '/ws/external/checkpoints/models/KITTI/no_orientation_prior/model.pt'

print('load model from: ' + test_model_path)
CVM_model.load_state_dict(torch.load(test_model_path))
CVM_model.to(device)
CVM_model.eval()


distance = []
distance_in_meters = []
longitudinal_error_in_meters = []
lateral_error_in_meters = []
orientation_error = []
angle_diff_list = []

start_time = time.time()

results = {}
for i, data in enumerate(demo_loader, 0):
    if i % 1000 == 0:
        print(i)
    sat, grd, gt, gt_with_ori, gt_orientation, orientation_angle, file_name = data
    grd = grd.to(device)
    sat = sat.to(device)

    logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)

    gt = gt.cpu().detach().numpy()
    orientation_angle = orientation_angle.cpu().detach().numpy()
    gt_orientation = gt_orientation.cpu().detach().numpy()
    heatmap = heatmap.cpu().detach().numpy()
    ori = ori.cpu().detach().numpy()
    for batch_idx in range(gt.shape[0]):
        orientation_from_north = orientation_angle[batch_idx] # degree
        current_gt = gt[batch_idx, :, :, :]
        loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape) # [1, x, y]
        current_pred = heatmap[batch_idx, :, :, :]
        loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
        pixel_distance = np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)
        distance.append(pixel_distance)
        meter_distance = pixel_distance * demo_set.meter_per_pixel
        distance_in_meters.append(meter_distance)

        gt2pred_from_north = np.arctan2(np.abs(loc_gt[2]-loc_pred[2]), np.abs(loc_gt[1]-loc_pred[1])) * 180 / math.pi # degree
        angle_diff = np.abs(orientation_from_north-gt2pred_from_north)
        angle_diff_list.append(angle_diff)

        pixel_distance_longitudinal = np.abs(np.cos(angle_diff * np.pi/180) * pixel_distance)
        pixel_distance_lateral = np.abs(np.sin(angle_diff * np.pi/180) * pixel_distance)
        longitudinal_error_in_meters.append(pixel_distance_longitudinal * demo_set.meter_per_pixel)
        lateral_error_in_meters.append(pixel_distance_lateral * demo_set.meter_per_pixel)

        cos_pred, sin_pred = ori[batch_idx, :, loc_pred[1], loc_pred[2]]
        if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <=1:
            a_acos_pred = math.acos(cos_pred)
            if sin_pred < 0:
                angle_pred = math.degrees(-a_acos_pred) % 360
            else:
                angle_pred = math.degrees(a_acos_pred)
            cos_gt, sin_gt = gt_orientation[batch_idx, :, loc_gt[1], loc_gt[2]]
            a_acos_gt = math.acos(cos_gt)
            if sin_gt < 0:
                angle_gt = math.degrees(-a_acos_gt) % 360
            else:
                angle_gt = math.degrees(a_acos_gt)
            orientation_error.append(np.min([np.abs(angle_gt-angle_pred), 360-np.abs(angle_gt-angle_pred)]))

        results[file_name[batch_idx]] = {}
        results[file_name[batch_idx]]['gt_loc'] = loc_gt[1:3] # [x, y]
        results[file_name[batch_idx]]['pred_loc'] = loc_pred[1:3] # [x, y]
        results[file_name[batch_idx]]['gt_ori'] = orientation_from_north - 90 # degree from east, clockwise
        results[file_name[batch_idx]]['pred_ori'] = gt2pred_from_north - 90 # degree from east, clockwise

        # results['y'].extend(loc_pred[2])
        # results['theta'].extend(orientation_angle)

# save results
import pickle
with open(f'/ws/external/ccvpe_results/CCVPE_KITTI_360_results_demo_{data_date}.pkl', 'wb') as f:
    pickle.dump(results, f)

# check inference time
end_time = time.time()
duration = (end_time - start_time)/len(demo_loader)

print('---------------------------------------')
print('Test 1 set')
print('Inference time: ', duration)
print(f'FPS: {1 / duration}')
print('mean localization error (m): ', np.mean(distance_in_meters))
print('median localization error (m): ', np.median(distance_in_meters))

print('---------------------------------------')
print('mean orientation error (degrees): ', np.mean(orientation_error))
print('median orientation error (degrees): ', np.median(orientation_error))

print('---------------------------------------')
longitudinal_error_in_meters = np.array(longitudinal_error_in_meters)
lateral_error_in_meters = np.array(lateral_error_in_meters)
orientation_error = np.array(orientation_error)
print('percentage of samples with lateral localization error under 1m, 3m, and 5m: ', np.sum(lateral_error_in_meters<1)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<3)/len(lateral_error_in_meters), np.sum(lateral_error_in_meters<5)/len(lateral_error_in_meters))
print('percentage of samples with longitudinal localization error under 1m, 3m, and 5m: ', np.sum(longitudinal_error_in_meters<1)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<3)/len(longitudinal_error_in_meters), np.sum(longitudinal_error_in_meters<5)/len(longitudinal_error_in_meters))
print('percentage of samples with orientation error under 1 degree, 3 degrees, and 5 degrees: ', np.sum(orientation_error<1)/len(orientation_error), np.sum(orientation_error<3)/len(orientation_error), np.sum(orientation_error<5)/len(orientation_error))
