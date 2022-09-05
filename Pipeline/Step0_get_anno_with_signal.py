# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import random
import cv2
from imageio import imread
import shutil
import pdb
from shapely import geometry
# label_dict = {'原始细胞（圆形）': 'a', '中幼，早幼细胞（椭圆形）': 'b', '晚幼细胞（豆芽形）': 'c',
#               '杆状核细胞（杆状）': 'd', '分叶核细胞（分叶状）': 'e', '其他细胞（不规则状）': 'f',
#               '死细胞': 'g', '自发荧光细胞': 'h', '非细胞成分': 'i', '气泡': 'j'}

# her2_label_dict = {'可计数细胞核（组织）': 'a'}

def get_match_segment_results(center_coords, labels, contours):
    signal_points = []
    signal_labels = []
    sample_num = center_coords.shape[0]
    poly_context = {'type': 'MULTIPOLYGON',
                    'coordinates': [[contours]]}
    # print('poly_context', poly_context)
    poly = geometry.shape(poly_context)
    for i in range(sample_num):
        each_coord = center_coords[i]
        point = geometry.Point(each_coord[0], each_coord[1])
        print('point', point)
        if point.within(poly):
            print('Yeah, I got points')
            signal_points.append(each_coord)
            signal_label = labels[i]
            signal_labels.append(signal_label)

    return signal_points, signal_labels

def get_per_anno(xy_coord, img_id, category, anno_id_check_list, all_signal_points, all_signal_labels, project):
    global category_id_dict
    if project == 'Mix':
        category_id_dict = {'a':1, 'b':2, 'c':3, 'f':4, 'g':5, 'h':6, 'i':7, 'j':8}
    if project == 'Her2':
        category_id_dict = {'a': 1}

    category_id = category_id_dict[category]
    per_anno_dict = {}
    instance_mask = xy_coord.flatten().tolist()
    area = cv2.contourArea(xy_coord)
    bbox = [min(xy_coord[:, 0]), min(xy_coord[:, 1]),
            max(xy_coord[:, 0]) - min(xy_coord[:, 0]), max(xy_coord[:, 1]) - min(xy_coord[:, 1])]
    signal_points, signal_labels = get_match_segment_results(all_signal_points, all_signal_labels, xy_coord)
    per_anno_dict['segmentation'] = [instance_mask]
    per_anno_dict['area'] = area
    per_anno_dict['iscrowd'] = 0
    per_anno_dict['image_id'] = img_id
    per_anno_dict['bbox'] = bbox
    per_anno_dict['category_id'] = category_id
    per_anno_dict['signal_points'] = signal_points
    per_anno_dict['signal_labels'] = signal_labels
    # print('ffhwafiweuhfioawehf aiwuhf ')

    anno_id = random.randint(0, 10000)
    while anno_id in anno_id_check_list:
        anno_id = random.randint(0, 10000)
    per_anno_dict['id'] = anno_id

    return per_anno_dict, anno_id, category_id, area

def get_per_image(size, file_name, img_id_list):
    per_image_dict = {}
    per_image_dict['file_name'] = file_name
    per_image_dict['height'] = size[0]
    per_image_dict['width'] = size[1]
    img_id = random.randint(0, 1000)
    while img_id in img_id_list:
        img_id = random.randint(0, 1000)
    per_image_dict['id'] = img_id
    return per_image_dict, img_id

def get_per_category(category, category_id, category_check_list):

    if category not in category_check_list:
        per_categories_dict = {}
        per_categories_dict['id'] = category_id
        per_categories_dict['name'] = category
        return per_categories_dict
    else:
        return None

def make_json_file(img_json_dict, project):
    all_img_files = img_json_dict.keys()

    image_list = []
    anno_list = []
    category_list = []
    #----------------------------------
    img_id_check_list = []
    anno_id_check_list = []
    category_check_list = []
    if project == 'Mix':
        category_count_dict = {'原始细胞（圆形）': 0, '中幼，早幼细胞（椭圆形）': 0,
                               '晚幼细胞（豆芽形）': 0,'其他细胞（不规则状）': 0,'死细胞': 0,
                               '自发荧光细胞': 0, '非细胞成分': 0, '气泡': 0}
        label_dict = {'原始细胞（圆形）': 'a', '中幼，早幼细胞（椭圆形）': 'b', '晚幼细胞（豆芽形）': 'c',
                  '其他细胞（不规则状）': 'f','死细胞': 'g', '自发荧光细胞': 'h', '非细胞成分': 'i', '气泡': 'j'}
    elif project == 'Her2':
        category_count_dict = {'可计数细胞核（组织）': 0}
        label_dict = {'可计数细胞核（组织）': 'a'}
    else:
        raise ValueError('Project name {} is invalid'.format(project))

    # ----------------------------------
    ## img loop
    for img_file in all_img_files:
        print('------------------------------------------------')
        print('img_file', img_file)
        img = imread(img_file)
        file_name = os.path.basename(img_file)
        size = (img.shape[0], img.shape[1])
        per_image_dict, img_id = get_per_image(size, file_name, img_id_check_list)
        img_id_check_list.append(img_id)
        image_list.append(per_image_dict)
        ## annotation loop
        json_file = img_json_dict[img_file]
        print('json_file', json_file)

        all_signal_points, all_signal_labels = signal_gt(json_file)

        with open(json_file, "r", encoding="gbk") as f:
            index_info = json.load(f)
            results = index_info['result']

            for result in results:
                path = result['path']
                remark = result['result']

                if remark in label_dict.keys():
                    category_count_dict[remark] += 1
                    # print('remark: ', remark)
                    x = path['x']
                    y = path['y']
                    category = label_dict[remark]
                    xy_coord = np.array([coord for coord in zip(x, y)], dtype=np.float32)
                    xy_coord = xy_coord.astype(dtype=int)
                    area = cv2.contourArea(xy_coord)

                    if area > 0:
                        per_anno_dict, anno_id, category_id, area = get_per_anno(xy_coord, img_id, category,
                                                                                 anno_id_check_list,
                                                                                 all_signal_points, all_signal_labels,
                                                                                 project=project)
                        anno_id_check_list.append(anno_id)
                        anno_list.append(per_anno_dict)
                        per_categories_dict = get_per_category(category, category_id, category_check_list)
                    else:
                        per_categories_dict = None

                    if per_categories_dict is not None:
                        category_check_list.append(per_categories_dict['name'])
                        category_list.append(per_categories_dict)
                        print('per_categories_dict: ', per_categories_dict)



    category_list.sort(key=lambda x: list(x.values())[0])
    print('category_list: ', category_list)
    assert len(category_list) == len(label_dict)
    json_dict = {}
    json_dict['images'] = image_list
    json_dict['annotations'] = anno_list
    json_dict['categories'] = category_list

    return json_dict, category_count_dict

def get_img_json(sample_root, img_root):
    img_json_dict = {}
    for sample_id in os.listdir(sample_root):
        # sample_id = '6985847'
        dirname = os.path.join(sample_root, sample_id)
        if os.path.isdir(dirname):
            for file_name in os.listdir(dirname):
                if file_name.endswith('.jpg') or file_name.endswith('.JPG'):
                    image_name = file_name
                    break
            image_path = os.path.join(dirname, image_name)
            shutil.copy(image_path, os.path.join(img_root, os.path.basename(image_path)))
            json_path = os.path.join(dirname, 'result.json')
            # print('image_path', image_path, '\n',
            #       'json_path', json_path, '\n',
            #       '---------------------------------')
            img_json_dict[image_path] = json_path
    print('Totally {} samples.'.format(len(img_json_dict)))
    return img_json_dict

class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def signal_gt(json_file):
    label_dict = {'红色信号': 0, '绿色信号': 1}
    count_dict = {'红色信号': 0, '绿色信号': 0}
    signal_points = []
    signal_labels = []
    with open(json_file, "r", encoding="gbk") as f:
        index_info = json.load(f)
        results = index_info['result']
        for result in results:
            path = result['path']
            remark = result['result']
            if remark in label_dict.keys():
                x = path['x'][0]
                y = path['y'][0]
                signal_label = label_dict[remark] + 1
                signal_points.append([int(x), int(y)])
                signal_labels.append(int(signal_label))
                count_dict[remark] += 1

    signal_points = np.array(signal_points)
    signal_labels = np.array(signal_labels)
    return signal_points, signal_labels


if __name__ == '__main__':
    # project = 'Mix'
    # sample_root = '/data2/Caijt/SoftTeacher/FISH_raw/1217_FISH_v5_Raw_95'
    # img_root = '/data2/Caijt/SoftTeacher/data/FISH_1102/95_images'
    # target_json_path = '/data2/Caijt/SoftTeacher/data/FISH_1102/annotations/instances_all_1217.json'
    project_name = 'Her2'
    sample_root = '/data2/Caijt/FISH_mmdet/data/FISH_raw/HER2_RAW/'
    # sample_root = '/data2/Caijt/SoftTeacher/FISH_raw/HER2_RAW'
    img_root = '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Her2_Images/'
    # img_root = '/data2/Caijt/SoftTeacher/data/FISH_1102/her2_images'

    target_json_path = '/data2/Caijt/SoftTeacher/data/FISH_1102/annotations/instances_all_0415_her2.json'

    img_json_dict = get_img_json(sample_root, img_root)
    json_dict, category_count_dict = make_json_file(img_json_dict, project = project_name)
    print('----------category_count_dict------------', '\n', category_count_dict)
    json_file = open(target_json_path, mode='w', encoding= 'utf-8')
    json.dump(json_dict, json_file, cls=NpEncoder, ensure_ascii=False)

    '''########### CHECK ANNOPOINTS #############'''
