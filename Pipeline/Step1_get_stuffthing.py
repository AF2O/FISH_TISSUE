#!/usr/bin/python

__author__ = 'Caijt'

import sys
sys.path.append('/data2/Caijt/FISH_mmdet/cocostuffapi/PythonAPI')
import os
from cocostuffapi.PythonAPI.pycocotools.cocostuffhelper import cocoSegmentationToPng
from pycocotools.coco import COCO
import cv2


def cocoSegmentationToPngDemo(dataTypeAnn, dataDir, annotation_folder, stuffthing_folder, IntermediateFolderName):
    '''Converts COCO segmentation .json files (GT or results) to one .png file per image.'''
    # Define paths
    annPath = '%s/%s/%s.json' % (dataDir, annotation_folder, dataTypeAnn)
    pngFolder = '%s/%s/%s' % (dataDir, stuffthing_folder, IntermediateFolderName)

    # Create output folder
    if not os.path.exists(pngFolder):
        os.makedirs(pngFolder)

    # Initialize COCO ground-truth API
    coco = COCO(annPath)
    imgIds = coco.getImgIds()

    # Convert each image to a png
    imgCount = len(imgIds)
    for i in range(0, imgCount):
        imgId = imgIds[i]
        imgName = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.jpg', '')
        print('Exporting image %d of %d: %s' % (i+1, imgCount, imgName))
        segmentationPath = '%s/%s.png' % (pngFolder, imgName)
        try:
            cocoSegmentationToPng(coco, imgId, segmentationPath)
            print('yeah, saved')
        except:
            raise Exception('Something wrong in the function: cocoSegmentationToPng')

    return

def image2grey(map_path, target_grey_root):
    os.makedirs(target_grey_root, exist_ok= True)
    file_names = os.listdir(map_path)
    for x in range(len(file_names)):
        img = cv2.imread(os.path.join(map_path, file_names[x]))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(target_grey_root, str(file_names[x])), img_gray)

    return
if __name__ == "__main__":

    dataTypeAnn = 'instances_all_0901_her2'
    dataDir = '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/'
    annotation_folder = 'Annotations'
    stuffthing_folder = 'Get_Stuffthing'
    IntermediateFolderName = 'intermediate_0901'
    TargetFolderName = 'stuffthingmaps_Her2_0901'
    cocoSegmentationToPngDemo(dataTypeAnn=dataTypeAnn, dataDir=dataDir,
                              annotation_folder = annotation_folder, stuffthing_folder = stuffthing_folder,
                              IntermediateFolderName=IntermediateFolderName)
    map_path = os.path.join(dataDir, stuffthing_folder, IntermediateFolderName)
    target_grey_root = os.path.join(dataDir, stuffthing_folder, TargetFolderName)
    image2grey(map_path, target_grey_root)


