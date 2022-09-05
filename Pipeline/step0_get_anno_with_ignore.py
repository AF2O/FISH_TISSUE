# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import random
import cv2
from imageio import imread
import shutil
import pdb
import sqlite3
from shapely import geometry
# label_dict = {'原始细胞（圆形）': 'a', '中幼，早幼细胞（椭圆形）': 'b', '晚幼细胞（豆芽形）': 'c',
#               '杆状核细胞（杆状）': 'd', '分叶核细胞（分叶状）': 'e', '其他细胞（不规则状）': 'f',
#               '死细胞': 'g', '自发荧光细胞': 'h', '非细胞成分': 'i', '气泡': 'j'}

# her2_label_dict = {'可计数细胞核（组织）': 'a'}
def db_connect(dataBase):
    """
    创建数据库链接，游标对象查询出的结果以数组形式返回
    :param dataBase:数据库文件路径
    :return: 数据库连接对象，游标对象
    """
    conn = sqlite3.connect(dataBase)
    cursor = conn.cursor()
    return conn, cursor

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
        # print('point', point)
        if point.within(poly):
            # print('Yeah, I got points')
            signal_points.append(each_coord)
            signal_label = labels[i]
            signal_labels.append(signal_label)

    return signal_points, signal_labels

def get_within_instance_ignore_mask(instance_mask, red_region_area_list):
    poly_context = {'type': 'MULTIPOLYGON',
                    'coordinates': [[instance_mask]]}
    poly = geometry.shape(poly_context)
    ignore_mask = []
    for i in range(len(red_region_area_list)):
        each_red_area = red_region_area_list[i]
        ref_polygon = geometry.Polygon(each_red_area)
        each_area_center = np.array(ref_polygon.centroid)
        # print('each_area_center', each_area_center)
        point = geometry.Point(each_area_center[0], each_area_center[1])
        if point.within(poly):
            # print('------------------------------- yeah ---------------------------------')
            ignore_mask.append(np.squeeze(each_red_area))
    return ignore_mask


def get_per_anno(xy_coord, img_id, category, anno_id_check_list, all_signal_points, all_signal_labels, red_region_area_list, project):
    global category_id_dict
    if project == 'Mix':
        category_id_dict = {'a':1, 'b':2, 'c':3, 'f':4, 'g':5, 'h':6, 'i':7, 'j':8}
    if project == 'Her2':
        category_id_dict = {'a': 1}

    category_id = category_id_dict[category]
    per_anno_dict = {}
    instance_mask = xy_coord.flatten().tolist()

    if not red_region_area_list:
        ignore_mask = []
    else:
        ignore_mask = get_within_instance_ignore_mask(xy_coord, red_region_area_list)

    print('ignore_mask', ignore_mask)

    area = cv2.contourArea(xy_coord)
    bbox = [min(xy_coord[:, 0]), min(xy_coord[:, 1]),
            max(xy_coord[:, 0]) - min(xy_coord[:, 0]), max(xy_coord[:, 1]) - min(xy_coord[:, 1])]
    signal_points, signal_labels = get_match_segment_results(all_signal_points, all_signal_labels, xy_coord)
    per_anno_dict['segmentation'] = [instance_mask]
    per_anno_dict['ignore_area'] = ignore_mask
    per_anno_dict['area'] = area
    per_anno_dict['iscrowd'] = 0
    per_anno_dict['image_id'] = img_id
    per_anno_dict['bbox'] = bbox
    per_anno_dict['category_id'] = category_id
    per_anno_dict['signal_points'] = signal_points
    per_anno_dict['signal_labels'] = signal_labels
    ignore_num = len(ignore_mask)
    if len(ignore_mask) > 0:
        print('---------------------ignore_mask num-----------------', ignore_num)

    anno_id = random.randint(0, 10000)
    while anno_id in anno_id_check_list:
        anno_id = random.randint(0, 10000)
    per_anno_dict['id'] = anno_id

    return per_anno_dict, anno_id, category_id, area, ignore_num

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

def make_json_file(img_json_dict, img_db_dict, project):
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

    table_name = 'Mark_label_fishTissue'

    # ----------------------------------
    ## img loop
    for img_file in all_img_files:

        # img_file = '/data2/Caijt/SoftTeacher/data/FISH_1102/her2_images/2020-03767E.jpg'
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
        if img_file in img_db_dict.keys():
            db_file = img_db_dict[img_file]
        else:
            db_file = None

        print('json_file', json_file)

        all_signal_points, all_signal_labels = signal_gt(json_file)

        if db_file is not None:
            conn, cursor = db_connect(db_file)
            marks = cursor.execute(
                "select position, groupId from {} where groupId is not null".format(table_name)).fetchall()
            image_list.append(per_image_dict)
            red_region_area_list = []
            for mark in marks:
                if mark:
                    # group_name = cursor.execute('select groupName from MarkGroup where id=?', (mark[1],)).fetchone()
                    remark = '红色团簇'
                    position = json.loads(mark[0])
                    x = position['x']
                    y = position['y']
                    red_region_xy_coord = np.array([coord for coord in zip(x, y)], dtype=np.float32)
                    red_region_xy_coord = red_region_xy_coord.astype(dtype=int)
                    red_region_area_list.append(red_region_xy_coord)
        else:
            red_region_area_list = []

        print('red_region_area_num', len(red_region_area_list))

        with open(json_file, "r", encoding="gbk") as f:
            index_info = json.load(f)
            results = index_info['result']
            ignore_total_num = 0
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
                        per_anno_dict, anno_id, category_id, area, ignore_num = get_per_anno(xy_coord, img_id, category,
                                                                                 anno_id_check_list,
                                                                                 all_signal_points, all_signal_labels,
                                                                                 red_region_area_list,
                                                                                 project=project)
                        anno_id_check_list.append(anno_id)
                        anno_list.append(per_anno_dict)
                        per_categories_dict = get_per_category(category, category_id, category_check_list)
                        ignore_total_num = ignore_total_num + ignore_num
                    else:
                        per_categories_dict = None

                    if per_categories_dict is not None:
                        category_check_list.append(per_categories_dict['name'])
                        category_list.append(per_categories_dict)
                        print('per_categories_dict: ', per_categories_dict)
            print('ignore_total_num', ignore_total_num)


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
            img_json_dict[os.path.join(img_root, os.path.basename(image_path))] = json_path
    print('Totally {} samples.'.format(len(img_json_dict)))
    return img_json_dict

def get_img_db(img_json_dict, db_files_root):
    img_db_dict = {}
    db_files = os.listdir(db_files_root)
    print('db_files', db_files)
    db_files = list(map(lambda x: os.path.basename(x)[:-3], db_files))
    print('db_files', db_files)

    for image_path in img_json_dict.keys():
        img_basename = os.path.basename(image_path)[:-4]
        print('img_basename',img_basename )
        if img_basename in db_files:
            db_path = os.path.join(db_files_root, img_basename+ '.db')
            print('db_path', db_path)
            assert os.path.exists(db_path)
            img_db_dict[image_path] = db_path

    print('Totally {} db samples.'.format(len(img_db_dict)))
    return img_db_dict


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

    project_name = 'Her2'
    # sample_root = '/data2/Caijt/SoftTeacher/FISH_raw/FISH_TISSUE_RAW_108/'
    # db_files_root = '/data2/Caijt/SoftTeacher/FISH_raw/db_anno/'
    # img_root = '/data2/Caijt/SoftTeacher/data/FISH_1102/her2_images'
    # target_json_path = '/data2/Caijt/SoftTeacher/data/FISH_1102/annotations/instances_all_0708_her2.json'

    sample_root = '/data2/Caijt/FISH_mmdet/data/FISH_raw/FISH_TISSUE_RAW_108/'
    db_files_root = '/data2/Caijt/FISH_mmdet/data/FISH_raw/db_anno/'
    img_root = '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Her2_Images/'
    target_json_path = '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Annotations/instances_all_0901_her2.json'


    img_json_dict = get_img_json(sample_root, img_root)
    img_db_dict = get_img_db(img_json_dict, db_files_root)
    json_dict, category_count_dict = make_json_file(img_json_dict, img_db_dict, project = project_name)
    print('----------category_count_dict------------', '\n', category_count_dict)
    json_file = open(target_json_path, mode='w', encoding= 'utf-8')
    json.dump(json_dict, json_file, cls=NpEncoder, ensure_ascii=False)

    '''########### CHECK ANNOPOINTS #############'''

    '--------------------------------------- one image for checking sanity--------------------------------------'

    # project_name = 'Her2'
    # # sample_root = '/data2/Caijt/SoftTeacher/FISH_raw/HER2_RAW'
    # sample_root = '/data2/Caijt/SoftTeacher/FISH_raw/FISH_TISSUE_RAW_1/'
    # db_files_root = '/data2/Caijt/SoftTeacher/FISH_raw/db_anno_1/'
    # img_root = '/data2/Caijt/SoftTeacher/data/FISH_1102/her_image_1/'
    # # target_json_path = '/data2/Caijt/SoftTeacher/data/FISH_1102/annotations/instances_all_0708_her2.json'
    # target_json_path = '/data2/Caijt/SoftTeacher/data/FISH_1102/annotations/instances_all_0624_her2_single_sample.json'
    #
    # img_json_dict = get_img_json(sample_root, img_root)
    # img_db_dict = get_img_db(img_json_dict, db_files_root)
    # # pdb.set_trace()
    # json_dict, category_count_dict = make_json_file(img_json_dict, img_db_dict, project = project_name)
    # print('----------category_count_dict------------', '\n', category_count_dict)
    # json_file = open(target_json_path, mode='w', encoding= 'utf-8')
    # json.dump(json_dict, json_file, cls=NpEncoder, ensure_ascii=False)
