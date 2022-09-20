import os
import glob
import cv2
from skimage import io
import numpy as np

if __name__ == '__main__':
    szl_root = '/data2/Caijt/FISH_mmdet/data/FISH_raw/817_FISH_example/'
    szl_target_root = '/data2/Caijt/FISH_mmdet/data/FISH_SZL/organized_JPG/'
    tif_list = glob.glob(os.path.join(szl_root, '*', '*', '*.tif'))

    for each_tif in tif_list:

        base_name = os.path.basename(each_tif)
        id_name = os.path.splitext(base_name)[0].split('_')[-1]
        second_level_dir = os.path.basename(os.path.dirname(each_tif))
        first_level_dir = os.path.basename(os.path.dirname(os.path.dirname(each_tif)))
        target_path = os.path.join(szl_target_root, first_level_dir + '_' + second_level_dir + '_' + str(id_name) + '.jpg')

        img = io.imread(each_tif)
        img = img / img.max()
        img = img * 255 - 0.0001
        img = img.astype(np.uint8)
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        bgr = cv2.merge([r, g, b])
        cv2.imwrite(target_path, bgr)




