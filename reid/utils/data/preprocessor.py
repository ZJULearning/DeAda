from __future__ import absolute_import
import os.path as osp
from .random_erasing import RandomErasing
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
# random flip
def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image
# random  HSV augmentation
def randomHueSaturationValue(image,  hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image
#random scale change
def randomShiftScaleRotate(image,
                           shift_limit=(-0.1, 0.1),
                           scale_limit=(-0.1, 0.1),
                           aspect_limit=(-0.1, 0.1),
                           rotate_limit=(-0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
    return image


class Preprocessor(object):
    def __init__(self, dataset,probability_re = 0.5,
                 sl=0.02, sh = 0.3, r1=0.4, mean=[0.4914, 0.4822, 0.4465],
                 root=None, transform=None, RE = False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.RE = RE
        self.transform = transform
        self.rand_erasing = RandomErasing(probability=probability_re,sl=sl,sh=sh,r1=r1,mean=mean)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index][:3]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        #if self.RE:
        #    img = self.rand_erasing(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid
