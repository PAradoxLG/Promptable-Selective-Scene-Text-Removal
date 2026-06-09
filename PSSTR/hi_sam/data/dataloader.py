# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import copy
import numbers
import sys

import numpy as np
import random
from copy import deepcopy

import torchvision.transforms
from skimage import io
import os
from glob import glob
from typing import Tuple, List, Optional
from collections.abc import Sequence
import json
from pycocotools import mask as mask_utils
import time
from tqdm import tqdm
import cv2
from PIL import Image


import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize, InterpolationMode
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, rotate, gaussian_blur
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler


#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>",flag," dataset ",i+1,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"],': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [
                datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"]
                for x in tmp_im_list
            ]
            print('-gt-',datasets[i]["name"],datasets[i]["gt_dir"],': ',len(tmp_gt_list))

        if(datasets[i]["no_text_im_dir"]==""):
            print('-no_text_im-', datasets[i]["name"], datasets[i]["no_text_im_dir"], ': ', 'No non-text image Found')
            tmp_no_text_im_list = []
        else:
            tmp_no_text_im_list = [
                datasets[i]["no_text_im_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["no_text_im_ext"]
                for x in tmp_im_list
            ]
            print('-no_text_im-',datasets[i]["name"],datasets[i]["no_text_im_dir"],': ',len(tmp_no_text_im_list))

        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "no_text_im_path":tmp_no_text_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "no_text_im_ext":datasets[i]["no_text_im_ext"],
                                "gt_ext":datasets[i]["gt_ext"],
                                "json_path": datasets[i].get('json_dir', None),
                                'word_path': [
                                    datasets[i]["word_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["word_ext"]
                                    for x in tmp_im_list
                                ] if datasets[i].get('word_dir') and datasets[i].get('word_ext') else None,
                                "word_ext": datasets[i].get('word_ext', None)
                                })

    return name_im_gt_list

def custom_collate_fn(batch_samples):
    imidx, image, label, shape = [], [], [], []
    word_masks, word_list, ori_im_path = [], [], []
    no_text_images = []
    for sample in batch_samples:
        imidx.append(sample['imidx'])
        image.append(sample['image'].unsqueeze(0))
        label.append(sample['label'].unsqueeze(0))
        shape.append(sample['shape'].unsqueeze(0))
        if sample.get('word_masks') is not None:
            word_masks.append(sample['word_masks'])
            word_list.append(sample['word_list'])
        ori_im_path.append(sample['ori_im_path'])
        if sample.get('no_text_images', None) is not None:
            no_text_images.append(torch.stack(sample['no_text_images']))
    return {
        'imidx': torch.as_tensor(imidx), 'image': torch.cat(image), 'label': torch.cat(label), 'shape': torch.cat(shape),
        'no_text_images': no_text_images,
        'word_masks': word_masks,
        'word_list': word_list,
        'ori_im_path': ori_im_path
        # 'paragraph_masks': para_masks, 'line_masks': line_masks,  'line2paragraph_index': line2para_idx
    }

def create_dataloaders(
        name_im_gt_list, my_transforms=[], batch_size=1, training=False, promptable=False, collate_fn=None
):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size >= 4:
        num_workers_ = 4
    if batch_size >= 8:
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform = transforms.Compose(my_transforms),
                promptable=promptable
            )
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        # sampler = DistributedSampler(gos_dataset)
        sampler = torch.utils.data.RandomSampler(gos_dataset)
        # sampler = torch.utils.data.SequentialSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
        dataloader = DataLoader(
            gos_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=num_workers_,
            pin_memory=True,
            prefetch_factor=6,
            collate_fn=collate_fn if promptable else None
        )
        gos_dataloaders = dataloader
        gos_datasets = gos_dataset
    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform=transforms.Compose(my_transforms),
                promptable=promptable,
                eval_ori_resolution=True
            )
            # sampler = DistributedSampler(gos_dataset, shuffle=False)
            sampler = torch.utils.data.SequentialSampler(gos_dataset)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_, collate_fn=collate_fn if promptable else None)
            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


class ResizeLongestSide_ToTensor(object):
    def __init__(self, target_length=1024):
        self.target_length = target_length

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]

    def __call__(self, sample):
        # image: np.array, [h, w, c]
        image, shape, label = sample['image'], sample['shape'], sample['label']
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        target_size = self.get_preprocess_shape(image.shape[1], image.shape[2], self.target_length)
        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), target_size, mode='bilinear'), dim=0)
        # lg20250330
        masks = sample.get('word_masks', None)
        if masks is not None:
            sample['word_masks'] = torch.from_numpy(masks).permute(2, 0, 1)
        # return {'image':image, 'shape':torch.tensor(target_size)}
        return {'image':image, 'shape':torch.tensor(target_size), 'label':sample['label'], 'no_text_images':sample['no_text_images'], 'word_masks':sample['word_masks']}


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        no_text_images = sample.get('no_text_images', None)
        masks = sample.get('word_masks', None)
        image = torch.from_numpy(image).permute(2, 0, 1)
        if no_text_images is not None:
            no_text_images = [torch.from_numpy(t).permute(2, 0, 1) for t in no_text_images]
        if len(label.shape) == 2:
            label = torch.from_numpy(label).unsqueeze(0)
        else:
            raise NotImplementedError
        sample['image'] = image
        sample['no_text_images'] = no_text_images
        sample['label'] = label
        if masks is not None:
            sample['word_masks'] = torch.from_numpy(masks).permute(2, 0, 1)
        return sample


class LargeScaleJitter(object):
    """
            implementation of large scale jitter from copy_paste
            https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
        """
    def __init__(self, output_size=1024, aug_scale_min=0.5, aug_scale_max=2.0):
        self.desired_size = output_size
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def __call__(self, sample):
        image, label, image_size = sample['image'], sample['label'], sample['shape']
        no_text_images = sample.get('no_text_images', None)
        masks = sample.get('word_masks', None)
        image_size = image_size.numpy()

        random_scale = np.random.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()
        scale = np.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().astype(np.int64)  # h, w

        scaled_image = cv2.resize(image, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
        if no_text_images is not None:
            scaled_no_text_images = [cv2.resize(t, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR) for t in no_text_images]
        scaled_label = cv2.resize(label, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))
        margin_h = max(scaled_size[0] - crop_size[0], 0)
        margin_w = max(scaled_size[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        scaled_image = scaled_image[crop_y1:crop_y2, crop_x1:crop_x2, :]
        if scaled_no_text_images is not None:
            scaled_no_text_images = [t[crop_y1:crop_y2, crop_x1:crop_x2, :] for t in scaled_no_text_images]
        scaled_label = scaled_label[crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.shape[0], 0)
        padding_w = max(self.desired_size - scaled_image.shape[1], 0)
        image = cv2.copyMakeBorder(scaled_image,0,padding_h,0,padding_w,cv2.BORDER_CONSTANT,value=(128,128,128))
        if scaled_no_text_images is not None:
            no_text_images = [cv2.copyMakeBorder(t,0,padding_h,0,padding_w,cv2.BORDER_CONSTANT,value=(128,128,128)) for t in scaled_no_text_images]
        label = cv2.copyMakeBorder(scaled_label,0,padding_h,0,padding_w,cv2.BORDER_CONSTANT,value=0)
        sample.update(image=image, label=label, no_text_images=no_text_images, shape=torch.tensor(image.shape[:2]))

        if masks is not None:
            masks = cv2.resize(masks, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]
            masks = masks[crop_y1:crop_y2, crop_x1:crop_x2, :]
            masks = cv2.copyMakeBorder(masks, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]
            sample['word_masks'] = masks
        return sample


class ColorJitter(object):
    def __init__(self, brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_params(self,
                    brightness: Optional[List[float]],
                    contrast: Optional[List[float]],
                    saturation: Optional[List[float]],
                    hue: Optional[List[float]]
                    ):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, sample):
        image = sample['image']  # np.array, hwc
        # image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)
        image = image.astype(np.float32)
        no_text_images = sample.get('no_text_images', None)

        if no_text_images is not None:
            # no_text_images = [torch.from_numpy(t).permute(2, 0, 1).unsqueeze(dim=0) for t in no_text_images]
            no_text_images = [t.astype(np.float32) for t in no_text_images]

        # fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
        #     self._get_params(self.brightness, self.contrast, self.saturation, self.hue)
        # for idx in fn_idx:
        #     if idx == 0 and brightness_factor is not None:
        #         image = adjust_brightness(image, brightness_factor)
        #         if no_text_images is not None:
        #             no_text_images = [adjust_brightness(t, brightness_factor) for t in no_text_images]
        #     elif idx == 1 and contrast_factor is not None:
        #         image = adjust_contrast(image, contrast_factor)
        #         if no_text_images is not None:
        #             no_text_images = [adjust_contrast(t, contrast_factor) for t in no_text_images]
        #     elif idx == 2 and saturation_factor is not None:
        #         image = adjust_saturation(image, saturation_factor)
        #         if no_text_images is not None:
        #             no_text_images = [adjust_saturation(t, saturation_factor) for t in no_text_images]
        #     elif idx == 3 and hue_factor is not None:
        #         image = adjust_hue(image, hue_factor)
        #         if no_text_images is not None:
        #             no_text_images = [adjust_hue(t, hue_factor) for t in no_text_images]

        # sample['image'] = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float32)
        # if no_text_images is not None:
        #     sample['no_text_images'] = [t.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float32) for t in no_text_images]
        return sample


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomRotate(object):
    def __init__(self, angle=180):
        self.angle = _setup_angle(angle, name="angle", req_sizes=(2, ))

    def apply(self, img, rm_img, bound_w, bound_h, interp=None, border_value=None):
        interp = interp if interp is not None else cv2.INTER_LINEAR
        return cv2.warpAffine(img, rm_img, (bound_w, bound_h), flags=interp, borderValue=border_value)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        no_text_images = sample.get('no_text_images', None)
        masks = sample.get('word_masks', None)

        h, w = image.shape[:2]
        angle = np.random.uniform(self.angle[0], self.angle[1])
        center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        bound_w, bound_h = np.rint(
            [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
        ).astype(int)
        rm_image = create_rotationmatrix(center, angle, bound_w, bound_h, offset=-0.5)

        image = self.apply(image, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, (128,128,128))
        if no_text_images is not None:
            no_text_images = [self.apply(t, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, (128,128,128)) for t in no_text_images]
        label = self.apply(label, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, 0)
        assert image.shape[:2] == label.shape[:2]
        sample['image'] = image
        sample['label'] = label
        if no_text_images is not None:
            sample['no_text_images'] = no_text_images
        sample['shape'] = torch.tensor(image.shape[:2])

        if masks is not None:
            masks = self.apply(masks, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, 0)
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]  # to avoid channel disappearing when there is only one mask
            sample['word_masks'] = masks

        return sample


def create_rotationmatrix(center, angle, bound_w, bound_h, offset=0):
    center_offset = (center[0] + offset, center[1] + offset)
    rm = cv2.getRotationMatrix2D(tuple(center_offset), angle, 1)
    rot_im_center = cv2.transform(center[None, None, :] + offset, rm)[0, 0, :]
    new_center = np.array([bound_w / 2, bound_h / 2]) + offset - rot_im_center
    rm[:, 2] += new_center
    return rm


def get_one_mask(vertices, w, h):
  mask = np.zeros((h, w), dtype=np.float32)
  mask = cv2.fillPoly(mask, [np.array(vertices)], [1])
  return mask


def get_word_mask(vertices, w, h):
  mask = np.zeros((h, w), dtype=np.float32)
  for ver in vertices:
    mask = cv2.fillPoly(mask, [np.array(ver)], [1])
  return mask

# 定义一个检查并修改小于0的值的函数
def check_and_modify(value):
    return max(int(value), 1) 

class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, promptable=False):
        # if hier_det is True, the json file for word, line, paragraph
        # detection should be loaded.
        self.transform = transform
        self.dataset = {}
        im_name_list = []  # image name
        im_path_list = []  # im path
        no_text_im_path_list = [] # no_text im path
        gt_path_list = []  # gt path

        assert len(name_im_gt_list) == 1
        name_im_gt_list = name_im_gt_list[0]
        im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list["im_ext"])[0] for x in name_im_gt_list["im_path"]])
        im_path_list.extend(name_im_gt_list["im_path"])
        no_text_im_path_list.extend(name_im_gt_list["no_text_im_path"])
        gt_path_list.extend(name_im_gt_list["gt_path"])

        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["no_text_im_path"] = no_text_im_path_list
        self.dataset["ori_no_text_im_path"] = deepcopy(no_text_im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset_name = name_im_gt_list["dataset_name"]

        self.promptable = promptable

        if promptable:
            word_path = name_im_gt_list.get('word_path', None)
            # print(word_path)
            assert word_path is not None, "Please check settings."
            print(f"Get {len(word_path)} word gt samples.")
            self.dataset["word_path"] = word_path
            

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        no_text_im_path = self.dataset["no_text_im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im_name = self.dataset["im_name"][idx]
        im = io.imread(im_path)
        no_text_im = io.imread(no_text_im_path)
        gt_ori = io.imread(gt_path)
        if 'TextSeg' in self.dataset_name:
            gt = (gt_ori == 100).astype(np.uint8) * 255
        elif 'COCO_TS' in self.dataset_name:
            gt = (gt_ori > 0).astype(np.uint8) * 255
        else:
            gt = (gt_ori > 127).astype(np.uint8) * 255  # for TotalText, HierText: 0 or 255
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]

        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)

        if len(no_text_im.shape) < 3:
            no_text_im = no_text_im[:, :, np.newaxis]
        if no_text_im.shape[2] == 1:
            no_text_im = np.repeat(no_text_im, 3, axis=2)
        if no_text_im.shape[2] > 3: # discard alpha channel
            no_text_im = no_text_im[:, :, :3]

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt.astype(np.float32),
            "shape": torch.tensor(im.shape[:2]),
        }

        if self.promptable:
            if 'FlickrST' in self.dataset_name:
                word_path = self.dataset["word_path"][idx]
                h, w = im.shape[:2]
                with open(word_path, 'r') as f:
                    word_lines = f.read().splitlines()
                
                word_per_image = 10
                word_num = len(word_lines)
                word_polys, word_list = [], []

                for i in range(word_num):
                    x1, y1, x2, y2, x3, y3, x4, y4, _word = word_lines[i].strip().split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(check_and_modify, [x1, y1, x2, y2, x3, y3, x4, y4])

                    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                    word_polys.append(pts)

                    if _word.startswith('"') and _word.endswith('"'): #删掉首尾引号
                        _word = _word.strip('"')
                    word_list.append(_word)

                select_word_idx = random.sample(range(word_num), min(word_per_image, word_num))
                select_word_idx.sort()
                masks = [get_one_mask(word_polys[w_idx], w, h) for w_idx in select_word_idx]

                # generate partial no_text_image
                no_text_images = []
                for mask in masks:
                    tmp_no_text_im = np.copy(im) # copy from image
                    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=3) # dilate mask
                    tmp_no_text_im[mask == 1] = no_text_im[mask == 1] # cover text region in image from no_text_im
                    # import pdb; pdb.set_trace()
                    no_text_images.append(tmp_no_text_im)

                masks = np.array(masks).transpose((1, 2, 0))
                word_list = [word_list[w_idx] for w_idx in select_word_idx]

                sample['word_masks'] = masks
                sample['word_list'] = word_list
                sample['no_text_images'] = no_text_images
            else:
                raise NotImplementedError


        if self.transform:
            sample = self.transform(sample)
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        if self.eval_ori_resolution:
            sample["ori_label"] = torch.unsqueeze(torch.from_numpy(gt), 0)
            if 'TextSeg' in self.dataset_name:
                ignore_mask = (gt_ori == 255).astype(np.uint8) * 255
                sample["ignore_mask"] = torch.unsqueeze(torch.from_numpy(ignore_mask), 0)
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample


train_transforms = [
    ColorJitter(),  # image: np.uint8->np.float32
    RandomRotate(),
    LargeScaleJitter(),
    ToTensor()
]

# eval_transforms = [
#     ResizeLongestSide_ToTensor(),
# ]
eval_transforms = [
    ColorJitter(),  # image: np.uint8->np.float32
    LargeScaleJitter(),
    ToTensor()
]