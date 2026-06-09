import sys

import numpy as np
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor
import glob
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
import pyclipper
import warnings
from visualize import visualization
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default='./pretrained_checkpoint/sam_tss_l_hiertext.pth',
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    
    parser.add_argument("--promptable", action='store_true',
                        help="If False, only text stroke segmentation.")
    parser.add_argument("--unimask_decoder_weight", type=str, default=r'work_dirs\2025-03-15__185422\199.pth',
                        help="The path to unimask_decoder weight.")

    parser.add_argument("--word_prompt", action='store_true',
                        help="If False, not support word prompt for segmentation")
    parser.add_argument("--word_embedding_weight", type=str, default="./pretrained_checkpoint/text_encoder.pth",
                        help="pretrained word decoder.")
    parser.add_argument('--visual', action='store_true', help='whether generate visualization of features or not.')

    parser.add_argument("--erase_mode", default=False, action='store_true',
                        help="If False, only segment text for prompts")

    parser.add_argument('--input_size', default=[1024,1024], type=list)

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')

    return parser.parse_args()


def patchify(image: np.array, patch_size: int=256):
    h, w = image.shape[:2]
    patch_list = []
    h_num, w_num = h//patch_size, w//patch_size
    h_remain, w_remain = h%patch_size, w%patch_size
    row, col = h_num + int(h_remain>0), w_num + int(w_remain>0)
    h_slices = [[r * patch_size, (r + 1) * patch_size] for r in range(h_num)]
    if h_remain:
        h_slices = h_slices + [[h - h_remain, h]]
    h_slices = np.tile(h_slices, (1, col)).reshape(-1, 2).tolist()
    w_slices = [[i * patch_size, (i + 1) * patch_size] for i in range(w_num)]
    if w_remain:
        w_slices = w_slices + [[w-w_remain, w]]
    w_slices = w_slices * row
    assert len(w_slices) == len(h_slices)
    for idx in range(0, len(w_slices)):
        # from left to right, then from top to bottom
        patch_list.append(image[h_slices[idx][0]:h_slices[idx][1], w_slices[idx][0]:w_slices[idx][1], :])
    return patch_list, row, col


def unpatchify(patches, row, col):
    # return np.array
    whole = [np.concatenate(patches[r*col : (r+1)*col], axis=1) for r in range(row)]
    whole = np.concatenate(whole, axis=0)
    return whole


def patchify_sliding(image: np.array, patch_size: int=512, stride: int=256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j+patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i+patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])

    return patch_list, h_slice_list, w_slice_list


def unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    assert len(ori_size) == 2  # (h, w)
    whole_logits = np.zeros(ori_size)
    assert len(patch_list) == len(h_slice_list)
    assert len(h_slice_list) == len(w_slice_list)
    for idx in range(len(patch_list)):
        h_slice = h_slice_list[idx]
        w_slice = w_slice_list[idx]
        whole_logits[h_slice, w_slice] += patch_list[idx]

    return whole_logits


def show_points(coords, ax, marker_size=200):
    ax.scatter(coords[0], coords[1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = color if color is not None else np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_res(masks, scores, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def show_hi_masks(masks, word_masks, input_points, filename, image, scores):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, (line_para_masks, word_mask, hi_score, point) in enumerate(zip(masks, word_masks, scores, input_points)):
        # line_mask = line_para_masks[0]
        # para_mask = line_para_masks[1]
        # show_mask(para_mask, plt.gca(), color=np.array([255 / 255, 144 / 255, 30 / 255, 0.5]))
        # show_mask(line_mask, plt.gca())
        word_mask = word_mask[0].astype(np.uint8)
        select_word = word_mask
        contours, _ = cv2.findContours(word_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        select_word = None
        for cont in contours:
            epsilon = 0.002 * cv2.arcLength(cont, True) # 计算弯曲度
            approx = cv2.approxPolyDP(cont, epsilon, True) # 将轮廓进行精简
            # print("before >> ", approx.shape)
            points = approx.reshape((-1, 2)) # 将轮廓转化为二维坐标
            # print("after  >> ", approx.shape)
            if points.shape[0] < 4: # 不能构成多边形
                continue
            pts = unclip(points, 0.5)
            if len(pts) != 1:
                continue
            pts = pts[0].astype(np.int32)
            if cv2.pointPolygonTest(pts, (int(point[0]), int(point[1])), False) >= 0: # 判断prompt的点是否在多边形范围内
                select_word = pts
                break
        if select_word is not None:
            point = point.astype(np.int32)
            # print(word_mask.shape, [point])
            word_mask = cv2.fillPoly(np.zeros(word_mask.shape), [pts], 1)
            show_mask(word_mask, plt.gca(), color=np.array([30 / 255, 255 / 255, 144 / 255, 0.5]))
        show_points(point, plt.gca())
        print(f'point {i}: word {hi_score[0]}')

    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_res(masks, scores, input_point, input_label, input_box, input_mask, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, plt.gca())
        if input_mask is not None:
            show_mask(input_mask, plt.gca(), random_color=True)
        
        print(f"Score: {score[0]:.3f}")
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score[0]:.3f}")
    plt.axis('off')
    plt.savefig(filename,bbox_inches='tight',pad_inches=-0.1)
    plt.close()


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p) # 根据点创建多边形对象
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded



if __name__ == '__main__':
    args = get_args_parser()
    hisam = model_registry[args.model_type](args)
    ### lg20250310
    target_lora_model = ['image_encoder.blocks.{}.Space_Adapter.D_fc1'.format(i) for i in range(hisam.image_encoder.depth)] + ['image_encoder.blocks.{}.Space_Adapter.D_fc2'.format(i) for i in range(hisam.image_encoder.depth)] + ['image_encoder.blocks.{}.MLP_Adapter.D_fc1'.format(i) for i in range(hisam.image_encoder.depth)] + ['image_encoder.blocks.{}.MLP_Adapter.D_fc2'.format(i) for i in range(hisam.image_encoder.depth)]
    lora_config = LoraConfig(target_modules=target_lora_model, r=8)
    lora_sam = get_peft_model(hisam, lora_config)
    if args.unimask_decoder_weight:
            unimask_decoder_path = args.unimask_decoder_weight
            with open(unimask_decoder_path, "rb") as f:
                param_dict = torch.load(f, map_location="cpu")
            dict_keys = param_dict.keys()
            if 'optimizer' in dict_keys or 'lr_scheduler' in dict_keys or 'epoch' in dict_keys:
                param_dict = param_dict['model']
                if 'image_encoder' in param_dict.keys():
                    image_encoder_dict = param_dict['image_encoder']
            info = lora_sam.image_encoder.load_state_dict(image_encoder_dict, strict=False)
            print(f'image_encoder matched info: {info}')

    # hisam.eval()
    # hisam.to(args.device)
    # predictor = SamPredictor(hisam)
    lora_sam.eval()
    lora_sam.to(args.device)
    predictor = SamPredictor(lora_sam)

    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    for path in tqdm(args.input, disable=not args.output):
        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = os.path.basename(path).split('.')[0] + '.png'
            out_filename = os.path.join(args.output, img_name)
        else:
            assert len(args.input) == 1
            out_filename = args.output

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        predictor.set_image(image)

        input_point, input_label, input_box = None, None, None
        input_mask = None
        input_word = None
        if args.promptable:
            # >>>> 点prompt >>>>
            # input_point = np.array([[125, 275]])  # for demo/img293.jpg
            # input_label = np.ones(input_point.shape[0])

            # input_point = np.array([[274, 260],[389, 250], [350, 288]])  # for demo/FIS_1.jpg
            # input_label = np.ones(input_point.shape[0])

            # input_point = np.array([[352, 209]])  # for demo/FIS_546.jpg
            # input_label = np.ones(input_point.shape[0])

            # input_point = np.array([[100, 350]])  # for demo/FIS_546.jpg
            # input_label = np.ones(input_point.shape[0])

            # >>>> 框prompt >>>>
            # input_box = np.array([[43,140,342,279],[353,174,621,291]]) # box for train/FIS_199.jpg
            # input_box = np.array([[305,271,428,302]]) # box for train/FIS_1.jpg


            # >>>> mask prompt >>>>
            # input_mask = np.zeros((h, w)) # mask for FIS_1.jpg
            # input_mask[288:299, 255:289] = 255


            # input_word = ['Do','feed','coyotes'] # word for FIS_1.jpg
            # input_word = ['FAST',] # word for FIS_1.jpg
            input_word = ['WASH']


            # >>>>>>>>>>>>>>> 特征可视化 >>>>>>>>>>>>>>>>>>>>>>>>
            if args.visual:
                visualization(os.path.basename(path).split('.')[0], predictor, hisam, input_point, input_label, input_box, input_mask)

            pr_mask, pr_iou, pr_word_mask = predictor.predict(
                multimask_output=False,
                promptable=True,
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                mask=input_mask,
                words=input_word,
            )
            # print(pr_mask.shape, pr_iou.shape)

            all_mask = None
            flag = True
            for i, mask in enumerate(pr_word_mask):
                if flag:
                    mask = mask.astype(np.uint8)
                    mask = np.squeeze(mask)
                    all_mask = mask
                    all_mask[mask == 1] = 255
                    all_mask[mask == 0] = 0
                    flag = False
                else:
                    mask = mask.astype(np.uint8)
                    mask = np.squeeze(mask)
                    all_mask[mask == 1] = 255 
                # cv2.imwrite('mask_' + str(i) + '.png', mask)
            cv2.imwrite('all_mask.png', all_mask)

            if input_point is not None:
                show_hi_masks(pr_mask, pr_word_mask, input_point, out_filename, image, pr_iou)
                # show_res(pr_mask,pr_iou, input_point, input_label, input_box, out_filename, image)
            if input_box is not None:
                print(out_filename)
                pr_mask[pr_mask == True] = False
                show_res_multi(pr_mask,pr_iou, input_point, input_label, input_box, out_filename, image)
            if input_mask is not None:
                show_res(pr_mask,pr_iou, input_point, input_label, input_box, input_mask > 0, out_filename, image)
            if input_word is not None:
                print(out_filename)
                show_res(pr_mask,pr_iou, input_point, input_label, input_box, input_mask, out_filename, image)
            # show_res(pr_word_mask, pr_iou, input_point, input_label, input_box, out_filename, image)
            # pr_word_mask = pr_word_mask.astype(np.uint8)
            # print(np.unique(pr_word_mask))
            # pr_word_mask[pr_word_mask == 1] = 255
            # pr_word_mask[pr_word_mask == 0] = 0
            # cv2.imwrite('pred_mask.png', np.squeeze(pr_word_mask))
        else:
            raise RuntimeError("--promptable must be set.")