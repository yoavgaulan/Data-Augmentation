
import numpy as np
import os
from tqdm import tqdm
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def aug(image, mask, image_name):
    segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
    ia.seed(1)
# Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Resize((0.4, 2.0)),
        iaa.CropToFixedSize(width=480, height=480),
        iaa.Sometimes(0.5,  iaa.MultiplyBrightness((0.5, 1.3))),
        iaa.Sometimes(0.1, iaa.PerspectiveTransform(scale=(0.01, 0.15))),
        iaa.Sometimes(0.4, iaa.ElasticTransformation(alpha=100, sigma=10)),
        iaa.Sometimes(0.2, iaa.BilateralBlur( d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))),
        iaa.Sometimes(0.2, iaa.pillike.EnhanceSharpness()),
        iaa.Sometimes(0.25, iaa.Affine(rotate=90)),  # rotate by 90 degrees (affects segmaps)
        iaa.Sometimes(0.25, iaa.Affine(rotate=180)),
        iaa.Sometimes(0.25, iaa.Affine(rotate=-90)),
        iaa.Fliplr(0.4),
        iaa.Flipud(0.4),
        # iaa.Sometimes(0.10, iaa.imgcorruptlike.Cpyontrast(severity=(1, 2))),
        # iaa.Sometimes(0.10, iaa.imgcorruptlike.Brightness(severity=(1, 5))),
        ], random_order=False)
    # Augment images and segmaps.
    images_aug = []
    segmaps_aug = []


    dir_out = '/home/yandex/MLW2021/roeeesquira/data_train_480'
    path_imgs = os.path.join(dir_out, 'imgs')
    path_masks = os.path.join(dir_out, 'mask')
    for i in range(2000):
        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        images_aug.append(images_aug_i)
        original_shape = segmaps_aug_i.arr.shape
        segmaps_aug_i = segmaps_aug_i.arr.reshape((original_shape[0], original_shape[1]))
        segmaps_aug.append(segmaps_aug_i)
        path_im_out = os.path.join(path_imgs, image_name + f'_{i}.png')
        path_mask_out = os.path.join(path_masks, image_name + f'_{i}.png')
        cv2.imwrite(path_im_out, images_aug_i)
        cv2.imwrite(path_mask_out, segmaps_aug_i)

    return images_aug, segmaps_aug

def save_im(dir_out, images_aug, segmaps_aug, image_name):
    path_imgs = os.path.join(dir_out, 'imgs')
    path_masks = os.path.join(dir_out, 'mask')
    for i in range(500):
        path_im_out = os.path.join(path_imgs, image_name + f'_{i}.png')
        path_mask_out = os.path.join(path_masks, image_name + f'_{i}.png')
        cv2.imwrite(path_im_out, images_aug[i])
        cv2.imwrite(path_mask_out, segmaps_aug[i])
    print('!!!!!!!!!')

def main():

    dir_im = '/home/yandex/MLW2021/yoavgaulan/orginal_data/imgs'
    dir_mask = '/home/yandex/MLW2021/yoavgaulan/orginal_data/mask'
    index = 0
    for im in tqdm(os.listdir(dir_im)):
        # if im != '6b.png':
        #     continue
        if im not in os.listdir(dir_mask):
            continue
        # if 'DS' in im:
        #     index+=1
        #     continue
        # if index == 0:
        #     continue
        image = os.path.join(dir_im, im)
        mask = os.path.join(dir_mask, im)

        image = cv2.imread(image)
        mask = cv2.imread(mask, 0)
        np.unique(mask)
        image_name = im.split('.')[0]
        images_aug, segmaps_aug = aug(image, mask, image_name)
        # save_im(dir_out, images_aug, segmaps_aug, image_name)


if __name__=='__main__':
    main()
