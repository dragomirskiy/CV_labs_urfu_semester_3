import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    light_orange = np.array((0, 169, 150))
    dark_orange = np.array((100, 255, 255))
    light_white = np.array((70, 0, 200))
    dark_white = np.array((145, 180, 255))

    orange_mask = cv2.inRange(img_hsv, light_orange, dark_orange)
    white_mask = cv2.inRange(img_hsv, light_white, dark_white)

    result_mask = orange_mask + white_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (79, 60))
    closing = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' # if args.is_train else 'test'
    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
