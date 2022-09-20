#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import sys
sys.path.append('/data2/Caijt/FISH_mmdet')
import asyncio
import glob
import os.path as osp
import os
from argparse import ArgumentParser
import torch
import numpy as np
import mmcv
import pandas as pd
import time
from shapely import geometry
from mmcv import Config
from mmdet.apis import inference_detector
import cv2
from imageio import imsave, imread
from mmdet.apis.inference import init_detector
import re
from typing import Union
# from ssod.apis.inference import init_detector, save_result
# from ssod.utils import patch_config
pattern = re.compile("\$\{[a-zA-Z\d_.]*\}")

def get_match_segment_results(center_coords, labels, contours, type, dist_threshold):
    red_list = []
    green_list = []
    sample_num = center_coords.shape[0]
    poly_context = {'type': 'MULTIPOLYGON',
                    'coordinates': [[contours]]}
    # print('poly_context', poly_context)
    poly = geometry.shape(poly_context)
    for i in range(sample_num):
        each_coord = center_coords[i]
        point = geometry.Point(each_coord[0], each_coord[1])
        if point.within(poly):
            signal_label = labels[i]
            if signal_label == 0:
                red_list.append(each_coord)
            if signal_label == 1:
                green_list.append(each_coord)

    if len(red_list) == 0 or len(green_list) == 0:
        yellow_list = []
    elif type == 'Amplification' or type == 'Deletion':
        yellow_list = []
    elif type == 'Fracture' or type == 'Fusion':
        red_list, green_list, yellow_list = match_fusion_signals(red_list, green_list, dist_threshold=dist_threshold)
    else:
        raise ValueError('The FISH type is not available')

    red_count = len(red_list)
    green_count = len(green_list)
    yellow_count = len(yellow_list)

    summary_str = ""
    if red_count > 0:
        summary_str = summary_str + '{}R'.format(red_count)
    if green_count > 0:
        summary_str = summary_str + '{}G'.format(green_count)
    if yellow_count > 0:
        summary_str = summary_str + '{}Y'.format(yellow_count)

    return red_list, green_list, yellow_list, summary_str

def IOU(pred_box, gt_box):
    x1, y1, x2, y2 = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
    x1_g, y1_g, x2_g, y2_g = gt_box[0], gt_box[1], gt_box[2], gt_box[3]

    shadow_x1 = max(x1, x1_g)
    shadow_y1 = max(y1, y1_g)
    shadow_x2 = max(min(x2, x2_g), shadow_x1)
    shadow_y2 = max(min(y2, y2_g), shadow_y1)

    bbox_w = shadow_x2 - shadow_x1
    bbox_h = shadow_y2 - shadow_y1

    intersection = bbox_h * bbox_w
    union = (x2 - x1) * (y2 - y1) + (x2_g - x1_g) * (y2_g - y1_g) - intersection
    small_area = min((x2 - x1) * (y2 - y1), (x2_g - x1_g) * (y2_g - y1_g))

    iou = intersection / union
    ots = intersection / small_area

    return iou, ots

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)

def match_fusion_signals(red_list, green_list, dist_threshold):
    yellow_list = []
    red_list_copy = red_list.copy()
    green_list_copy = green_list.copy()
    for i, red_signal in enumerate(red_list_copy):
        for j, green_signal in enumerate(green_list_copy):
            distance = ((red_signal[0] - green_signal[0]) ** 2 + (red_signal[1] - green_signal[1]) ** 2) ** 0.5
            if distance < dist_threshold:
                yellow_coord = np.array([int((red_signal[0] + green_signal[0]) * 0.5), int((red_signal[1] + green_signal[1]) * 0.5)])
                yellow_list.append(tuple(yellow_coord))
                try:
                    removearray(red_list, red_signal)
                except:
                    pass
                try:
                    removearray(green_list, green_signal)
                except:
                    pass

    return red_list, green_list, yellow_list

def get_match_bbox_results(center_coords, labels, bbox, type, dist_threshold):
    red_list = []
    green_list = []
    sample_num = center_coords.shape[0]
    x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    for i in range(sample_num):
        each_coord = center_coords[i]
        x, y = each_coord[0], each_coord[1]
        if x > x1 and x < x2:
            if y > y1 and y < y2:
                signal_label = labels[i]
                if signal_label == 0:
                    red_list.append(each_coord)
                if signal_label == 1:
                    green_list.append(each_coord)

    if len(red_list) == 0 or len(green_list) == 0:
        yellow_list = []
    elif type == 'Amplification' or type == 'Deletion':
        yellow_list = []
    elif type == 'Fracture' or type == 'Fusion':
        red_list, green_list, yellow_list = \
            match_fusion_signals(red_list, green_list, dist_threshold=dist_threshold)
    else:
        raise ValueError('The FISH type is not available')

    red_count = len(red_list)
    green_count = len(green_list)
    yellow_count = len(yellow_list)

    summary_str = ""
    if red_count > 0:
        summary_str = summary_str + '{}R'.format(red_count)
    if green_count > 0:
        summary_str = summary_str + '{}G'.format(green_count)
    if yellow_count > 0:
        summary_str = summary_str + '{}Y'.format(yellow_count)

    return red_list, green_list, yellow_list, summary_str

def get_all_pred_points(signal_result):
    all_center_coords = []
    all_center_labels = []
    # signal_result = np.squee
    # print('########### signal_result here ##############', len(signal_result), len(signal_result[0]),
    #       signal_result[0][0].shape)
    for roi_signals in signal_result:
        center_coords = roi_signals[:, :2]
        center_labels = roi_signals[:, 2]
        # print('########### center_coords, center_labels ##############', center_coords, center_labels)
        for center_coord, center_label in zip(list(center_coords), list(center_labels)):
            center_label = int(center_label)
            if center_label == 1 or center_label == 2:
                all_center_coords.append(center_coord)
                all_center_labels.append(center_label)

    return all_center_coords, all_center_labels

def generate_bbox_summary(CELL_OR_TISSUE, result, type, dist_threshold,
                          score_thr, overlap_thr, ots_thr):
    if CELL_OR_TISSUE == 'CELL':
        class_name_dict = {0: "Primordial cell", 1: "Promyelocytic cells", 2: "Late immature cells",
                           3: "Other cell", 4: "Dead cell", 5: "Autofluorescent cells",
                           6: "Non-cellular components", 7:"bubble"}
        num_classes = 8
    else:
        class_name_dict = {0: "Nucleus"}
        num_classes = 1

    bbox_summary_dict = {}
    bbox_id = 0
    if isinstance(result, tuple):
        bbox_pred, segm_result, signal_result = result
        signal_result = signal_result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_pred, segm_result, signal_result = result, None, None

    red_count_all = 0
    green_count_all = 0
    cell_count = 0

    for class_idx in range(num_classes):
        class_name = class_name_dict[class_idx]
        bboxes = bbox_pred[class_idx]
        segments = segm_result[class_idx]
        for i, bbox in enumerate(bboxes):
            bbox_id += 1
            bbox_score = bbox[4]
            segment = segments[i]
            roi_signals = signal_result[i]

            if bbox_score < score_thr:
                continue
            if segment is not None:
                segment = np.uint8(segment * 255)
                ret, thresh = cv2.threshold(segment, 127, 255, 0)
                cv_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = np.array(
                    [np.array(cv_contour).squeeze() for cv_contour in cv_contours[0] if len(cv_contour) > 0])
                if len(contours) > 2:
                    red_list = []
                    green_list = []
                    yellow_list = []
                    center_coords = roi_signals[:, :2]
                    center_labels = roi_signals[:, 2]
                    for center_coord, center_label in zip(list(center_coords), list(center_labels)):
                        center_label = int(center_label)
                        if center_label == 1:
                            red_list.append(center_coord)
                        if center_label == 2:
                            green_list.append(center_coord)
                    summary_str = ""
                    red_count = len(red_list)
                    green_count = len(green_list)
                    if red_count > 0:
                        summary_str = summary_str + '{}R'.format(red_count)
                    if green_count > 0:
                        summary_str = summary_str + '{}G'.format(green_count)

                    area = (max(contours[:, 0]) - min(contours[:, 0])) * (max(contours[:, 1]) - min(contours[:, 1]))
                    if area > 300:
                        bbox_dict = {}
                        bbox_dict['bbox_id'] = bbox_id
                        bbox_dict['class_idx'] = 0
                        bbox_dict['class_name'] = class_name
                        bbox_dict['bbox'] = bbox
                        bbox_dict['area'] = area
                        # print('area', area)
                        bbox_dict['segment'] = contours
                        bbox_dict['show_segment'] = contours.reshape(-1, 1, 2)
                        bbox_dict['score'] = bbox_score
                        bbox_dict['summary'] = summary_str
                        bbox_dict['coords'] = {'red': red_list, 'green': green_list, 'yellow': yellow_list}

                        overlap = False
                        for record_idx in bbox_summary_dict.keys():
                            record = bbox_summary_dict[record_idx]
                            record_bbox = record['bbox']
                            iou, ots = IOU(record_bbox, bbox)
                            if iou > overlap_thr:
                                overlap = True
                                record_score = record['score']
                                if bbox_score > record_score:
                                    bbox_summary_dict[record_idx] = bbox_dict
                            if ots > ots_thr:
                                overlap = True
                                record_area = record['area']
                                if area < record_area:
                                    bbox_summary_dict[record_idx] = bbox_dict
                        #
                        if overlap == False:
                            bbox_summary_dict[bbox_id] = bbox_dict
                            red_count = len(bbox_dict['coords']['red'])
                            green_count = len(bbox_dict['coords']['green'])
                            if red_count > 0 and green_count > 0:
                                red_count_all += red_count
                                green_count_all += green_count
                                cell_count += 1

    return bbox_summary_dict, center_coords, center_labels


def generate_final_results(bbox_summary_dict):
    red_coords_list = []
    green_coords_list = []
    cell_boundary_list = []

    for idx in bbox_summary_dict.keys():
        bbox_dict = bbox_summary_dict[idx]
        contours = bbox_dict['segment']
        signal_dict =bbox_dict['coords']
        area = bbox_dict['area']
        red_coords = signal_dict['red']
        green_coords = signal_dict['green']

        if len(red_coords) > 0:
            red_coords_list.extend(red_coords[:])
        if len(green_coords) > 0:
            green_coords_list.extend(green_coords[:])
        cell_boundary_list.append(contours)

    return cell_boundary_list, red_coords_list, green_coords_list

def prediction_show(img_, bbox_summary_dict, instance_color_dict):

    color_dict = {"red": (255, 0, 0), "green": (0, 255, 0), "yellow": (255, 255, 0)}
    img = img_.copy()
    radius = 15
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in bbox_summary_dict.keys():
        bbox_dict = bbox_summary_dict[idx]
        bbox = bbox_dict['bbox']
        contours = bbox_dict['show_segment']
        class_name = bbox_dict['class_name']
        summary_str = bbox_dict['summary']
        signal_dict =bbox_dict['coords']
        area = bbox_dict['area']
        red_coords = signal_dict['red']
        green_coords = signal_dict['green']
        yellow_coords = signal_dict['yellow']
        x1, x2, y1, y2 = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
        cv2.drawContours(img, [contours], 0, instance_color_dict[class_name], 2)
        ## draw signals
        for red_coord in red_coords:
            img = cv2.circle(img, (int(red_coord[0]), int(red_coord[1])), radius, color_dict["red"], thickness)
        for green_coord in green_coords:
            img = cv2.circle(img, (int(green_coord[0]), int(green_coord[1])), radius, color_dict["green"], thickness)
        for yellow_coord in yellow_coords:
            img = cv2.circle(img, (int(yellow_coord[0]), int(yellow_coord[1])), radius, color_dict["yellow"], thickness)
        ## draw summary
        if summary_str != "":
            summary = '('+summary_str + ')'
            img = cv2.putText(img, summary, (x1, y2), font, 1, (255, 255, 255), 2)

    return img

def filter_bbox(bbox_summary_dict):
    bbox_summary_dict_copy = bbox_summary_dict.copy()
    final_bbox_summary_dict = {}
    for bbox_key, bbox_dict in bbox_summary_dict_copy.items():
        signal_dict = bbox_dict['coords']
        red_coords = signal_dict['red']
        green_coords = signal_dict['green']
        if len(green_coords) >= 1 and len(red_coords) >= 1:
            final_bbox_summary_dict[bbox_key] = bbox_dict
        # if len(green_coords) <= 1 or len(red_coords) <= 1:
        #     del bbox_summary_dict[bbox_key]
    return final_bbox_summary_dict



def save_both_results(CELL_OR_TISSUE, image_path, pred, out_file_path,
                      FISH_type, dist_threshold, score_thr, overlap_thr, ots_thr):
    if CELL_OR_TISSUE == 'TISSUE':
        # Nucleus： 金黄色
        instance_color_dict = {"Nucleus": (255, 255, 0)}
    elif CELL_OR_TISSUE == 'CELL':
        # Primordial cell： 金黄色， # Promyelocytic cells：橙红色， # Late immature cells：绿色，
        # Other cell：红色，# Dead cell：紫色， # Autofluorescent cells：浅蓝色，
        # Non - cellular components：灰色， # bubble：浅绿色
        instance_color_dict = {"Primordial cell": (255, 255, 0), "Promyelocytic cells": (255, 69, 0),
                               "Late immature cells": (0, 255, 0),
                               "Other cell": (255, 0, 0), "Dead cell": (128, 0, 255),
                               "Autofluorescent cells": (0, 128, 255),
                               "Non-cellular components": (128, 128, 128), "bubble": (128, 255, 0)}
    else:
        raise ValueError('You can only choose CELL or TISSUE for calculation')


    image = imread(image_path)
    bbox_start_time = time.time()
    bbox_summary_dict, center_coords, center_labels = generate_bbox_summary(CELL_OR_TISSUE, pred, type=FISH_type, dist_threshold=dist_threshold,
                                              score_thr=score_thr, overlap_thr=overlap_thr, ots_thr = ots_thr)
    print('generate_bbox_summary time', time.time() - bbox_start_time)
    show_start_time = time.time()
    # bbox_summary_dict = filter_bbox(bbox_summary_dict)
    final_image = prediction_show(image, bbox_summary_dict, instance_color_dict)
    imsave(out_file_path, final_image)
    print('imsave time', time.time() - show_start_time)

    cell_boundary_list, red_coords_list, green_coords_list = generate_final_results(bbox_summary_dict)
    return cell_boundary_list, red_coords_list, green_coords_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--CELL_OR_TISSUE",
                        # default="CELL",
                        default="TISSUE",
                        help="CELL_OR_TISSUE")
    parser.add_argument("--img_dir",
                        # default='/data2/Caijt/FISH_mmdet/data/FISH_raw/szl_test/',
                        # default="/data2/Caijt/FISH_mmdet/data/FISH_raw/HER2_TEST_0406/2021/",
                        default="/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Her2_Images_1/",
                        help="Image file")
    parser.add_argument("--output", type=str,
                        default='/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Test_Results/HER2_TEST_0920_single/',
                        help="specify the directory to save visualization results.")

    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--dist_threshold", type = float, default= 15, help="Distance threshold for fusion recognition")
    parser.add_argument("--FISH_type", default="Amplification", help="Amplification, Deletion, Fracture, Fusion")

    parser.add_argument("--score-thr", type= float, default=0.3, help="bbox score threshold")
    parser.add_argument("--overlap-thr", type=float, default=0.4, help="bbox score threshold")
    parser.add_argument("--ots-thr", type=float, default=0.85, help="3 bbox overlap threshold")

    args = parser.parse_args()
    return args

def get_value(cfg: dict, chained_key: str):
    keys = chained_key.split(".")
    if len(keys) == 1:
        return cfg[keys[0]]
    else:
        return get_value(cfg[keys[0]], ".".join(keys[1:]))

def resolve(cfg: Union[dict, list], base=None):
    if base is None:
        base = cfg
    if isinstance(cfg, dict):
        return {k: resolve(v, base) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [resolve(v, base) for v in cfg]
    elif isinstance(cfg, tuple):
        return tuple([resolve(v, base) for v in cfg])
    elif isinstance(cfg, str):
        # process
        var_names = pattern.findall(cfg)
        if len(var_names) == 1 and len(cfg) == len(var_names[0]):
            return get_value(base, var_names[0][2:-1])
        else:
            vars = [get_value(base, name[2:-1]) for name in var_names]
            for name, var in zip(var_names, vars):
                cfg = cfg.replace(name, str(var))
            return cfg
    else:
        return cfg

def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir

def patch_config(cfg):

    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)
    # enable environment variables
    setup_env(cfg)
    return cfg

if __name__ == "__main__":

    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    init_start_time = time.time()
    model = init_detector(cfg, args.checkpoint, device=args.device)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    print('init time', time.time() - init_start_time)
    # Do calculate
    imgs = []
    img_dir = args.img_dir
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        img_full_name = os.path.join(img_dir, img_name)
        imgs.append(img_full_name)
    empty_summary_dict = {'病理号': [], 'HER2': [], 'CEP17': [], 'HER2/CEP17': [], '细胞核总数': []}
    summary_dict = empty_summary_dict.copy()
    for img in imgs:
        img_start_time = time.time()
        result = inference_detector(model, img)
        print('inference time', time.time() - img_start_time)
        os.makedirs(args.output, exist_ok= True)
        out_file_path = os.path.join(args.output, os.path.basename(img))
        cell_boundary_list, red_coords_list, green_coords_list = \
            save_both_results(CELL_OR_TISSUE=args.CELL_OR_TISSUE,
                              image_path=img, pred=result,
                              out_file_path=out_file_path, FISH_type=args.FISH_type,
                              dist_threshold=args.dist_threshold, score_thr=args.score_thr,
                              overlap_thr=args.overlap_thr, ots_thr=args.ots_thr)
        print(f"Save results to {out_file_path}")
        summary_dict['病理号'].append(os.path.basename(img)),
        summary_dict['HER2'].append(len(red_coords_list)),
        summary_dict['CEP17'].append(len(green_coords_list)),
        if len(green_coords_list) == 0:
            summary_dict['HER2/CEP17'].append(len(red_coords_list) / 1e-8)
        else:
            summary_dict['HER2/CEP17'].append(len(red_coords_list) / len(green_coords_list))
        summary_dict['细胞核总数'].append(len(cell_boundary_list)),
        print('per image time all', time.time() - img_start_time)
        print('-----------------------------------------')
        # pdb.set_trace()
    summary_save_path = os.path.join(args.output, 'summary.csv')
    df = pd.DataFrame(summary_dict)
    columns = list(empty_summary_dict.keys())
    df.to_csv(summary_save_path, columns=columns, index=False)
