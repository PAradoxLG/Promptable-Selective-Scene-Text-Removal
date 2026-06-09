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
from dinet.model_util import create_decoder
from erase_predictor import ErasePredictor
import glob
from tqdm import tqdm
import random
from PIL import Image
from shapely.geometry import Polygon
import pyclipper
import warnings
from visualize import visualization
from skimage import io
from scipy.ndimage import label as findlabel
from peft import LoraConfig, get_peft_model
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./output_result',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default='./pretrained_checkpoint/sam_tss_l_hiertext.pth',
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="The device to run generation on.")
    parser.add_argument("--prompt_type", type=str, default="word",
                        help="The prompt type.")
    parser.add_argument("--mask_path", type=str, default="./datasets/FlickrST/test/word_box",
                        help="mask_path")
    parser.add_argument("--label_path", type=str, default="./datasets/FlickrST/test/label",
                        help="label path.")
    
    parser.add_argument("--promptable", action='store_true',
                        help="If False, only text stroke segmentation.")
    parser.add_argument("--unimask_decoder_weight", type=str, default=r'work_dirs\2025-03-15__185422\199.pth',
                        help="The path to unimask_decoder weight.")

    parser.add_argument("--word_prompt", action='store_true',
                        help="If False, not support word prompt for segmentation")
    parser.add_argument("--word_embedding_weight", type=str, default="./pretrained_checkpoint/text_encoder.pth",
                        help="pretrained word decoder.")
    parser.add_argument('--visual', action='store_true', help='whether generate visualization of features or not.')

    parser.add_argument("--erase_mode", default=True, action='store_true',
                        help="If False, only segment text for prompts")
    parser.add_argument("--erase_decoder_weight", type=str, default=r'work_dirs\2025-04-05__212032\best.pth',
                        help="The path to erasde_decoder weight.")

    # >>>>>>>>>>>>>>>>>> erase decoder params >>>>>>>>>>>>>>>>>>>>>>>
    parser.add_argument('--decoder', type=str, default='dualhelix')
    parser.add_argument('--Block', type=str, default='cnn')
    parser.add_argument('--InterBlock', type=str, default='cnn')
    parser.add_argument('--SegAux', type=str, default='sim_ffn')
    parser.add_argument('--SegAuxList', nargs='+')
    parser.add_argument('--RMAux', type=str, default='sim_ffn')
    parser.add_argument('--RMAuxList', nargs='+')
    parser.add_argument('--dec_num_block', nargs='+',type=int, default=[1, 2, 3])
    parser.add_argument('--aux_num_blocks', nargs='+',type=int, default=[2, 3, 3])
    # parser.add_argument('--SegAux', type=str, default='abs_diff')
    # parser.add_argument('--SegAuxList', nargs='+')
    # parser.add_argument('--RMAux', type=str, default='abs_diff')
    # parser.add_argument('--RMAuxList', nargs='+')
    # parser.add_argument('--dec_num_block', nargs='+',type=int, default=[1, 2, 2])
    # parser.add_argument('--aux_num_blocks', nargs='+',type=int, default=[2, 2, 2])
    parser.add_argument('--channels', nargs='+',type=int, default=[64, 128, 256])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    parser.add_argument('--input_size', default=[1024,1024], type=list)

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')

    return parser.parse_args()

def find_overlapping_regions(mask1, mask2):
    """
    输入两个形状为 (h, w, 1) 的二值掩码，返回所有重叠的连通区域信息。
    返回值格式：列表，每个元素为字典，包含原区域标签、重叠区域坐标和掩码。
    """
    # 确保输入为二维数组
    mask1 = mask1.squeeze()
    mask2 = mask2.squeeze()
    
    # 连通区域标注
    labeled1, num_regions1 = findlabel(mask1)
    labeled2, num_regions2 = findlabel(mask2)
    
    overlapping_regions = []
    
    # 遍历第一个掩码的每个区域
    for i in range(1, num_regions1 + 1):
        # 获取区域i的坐标
        rows, cols = np.where(labeled1 == i)
        if len(rows) == 0:
            continue
        
        # 在第二个掩码中查找覆盖的标签
        j_labels = labeled2[rows, cols]
        unique_j = np.unique(j_labels[j_labels != 0])  # 排除背景0
        
        for j in unique_j:
            # 计算区域i和区域j的交集
            region_i = (labeled1 == i)
            region_j = (labeled2 == j)
            overlap = region_i & region_j
            
            # 标注交集中的连通区域
            labeled_overlap, num_overlap = findlabel(overlap)
            for k in range(1, num_overlap + 1):
                overlap_mask = (labeled_overlap == k)
                coords = np.where(overlap_mask)
                overlapping_regions.append({
                    'mask1_label': i,
                    'mask2_label': j,
                    'coords': (coords[0], coords[1]),  # 全局坐标
                    'mask': overlap_mask                # 二值掩码
                })
    
    return overlapping_regions

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
    ax.scatter(coords[0], coords[1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)


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
    ### lg20250325
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
    hisam.eval()
    hisam.to(args.device)

    erase_model = create_decoder(args.decoder, args.Block, args.InterBlock,
                                   args.SegAux, args.RMAux,
                                   args.dec_num_block, args.aux_num_blocks,
                                   args.channels)
    
    if args.erase_decoder_weight:
        with open(args.erase_decoder_weight, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        dict_keys = state_dict.keys()
        if 'optimizer' in dict_keys or 'lr_scheduler' in dict_keys or 'epoch' in dict_keys:
            param2_dict = state_dict['model']
        del state_dict
        info1 = erase_model.load_state_dict(param2_dict['erase_decoder'], strict=False)
        print(f'erase_model matched info: {info1}')
        info2 = lora_sam.image_encoder.load_state_dict(param2_dict['image_encoder'], strict=False)
        print(f'image_encoder matched info: {info2}')
        info3 = lora_sam.unimask_decoder.load_state_dict(param2_dict['unimask_decoder'], strict=False)
        print(f'unimask_decoder matched info: {info3}')
    
    lora_sam.eval()
    lora_sam.to(args.device)

    erase_model.eval()
    erase_model.to(args.device)

    # predictor = ErasePredictor(hisam, erase_model)
    predictor = ErasePredictor(lora_sam, erase_model)

    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    prompt_type = 'mask_vis'
    args.output = os.path.join(args.output, prompt_type+'/')

    psnr_sum = 0
    psnr_num = 0
    psnr_sum2 = 0
    psnr_num2 = 0
    mae_sum = 0
    SSIM_sum = 0

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

        # print(image.shape) # (427, 640, 3)
        # print(type(image)) # <class 'numpy.ndarray'>

        predictor.set_image(image)

        input_point, input_label, input_box = None, None, None
        input_mask = None
        input_word = None
        flag = True
        if args.promptable and args.erase_mode:
            # input_word = ['South'] 
            basename = os.path.basename(path).split('.')[0]
            m_path = os.path.join(args.mask_path, basename + '.txt')
            l_path = os.path.join(args.label_path, basename + '.jpg')
            label = np.array(Image.open(l_path))[:, :, :3]

            fmask = open(m_path, 'r')
            lines = fmask.readlines()
            # sample = random.sample(range(0, len(lines)), 1)
            # line = lines[sample[0]].split(',')[:-1]
            gt_mask = np.zeros((h, w)) 
            for line in lines:
                read_word = line.split(',')[-1]
                line = line.split(',')[:-1]
                linex = line[::2]
                liney = line[1::2]
                linex = [int(float(i)) for i in linex]
                liney = [int(float(i)) for i in liney]
                gt_mask[min(liney):max(liney), min(linex):max(linex)] = 1
                if prompt_type == 'mask_vis':
                    input_mask = np.zeros((h, w))
                    lenX = max(linex) - min(linex)
                    lenY = max(liney) - min(liney)
                    deltx = int(lenX/6)
                    delty = int(lenY/6)
                    # input_mask[min(liney)+delty:max(liney)-delty, min(linex)+deltx:max(linex)-deltx] = 255
                    # input_mask[min(liney)-delty:max(liney)+delty, min(linex)-deltx:max(linex)+deltx] = 255
                    input_mask[min(liney):max(liney), min(linex):max(linex)] = 255
                elif prompt_type == 'point_vis':
                    C_X = (max(linex) + min(linex))/2
                    C_Y = (max(liney) + min(liney))/2
                    input_point = np.array([[C_X, C_Y]])
                    input_label = np.ones(input_point.shape[0])
                elif prompt_type == 'box_vis':
                    input_box = np.array([[min(linex),min(liney),max(linex),max(liney)]])
                elif prompt_type == 'word_vis':
                    input_word = [read_word]
                else:
                    raise RuntimeError("prompt_type != 'mask_vis'.")

                pr_mask, pr_iou, pr_word_mask, image_hr, mask_hr = predictor.predict(
                    multimask_output=False,
                    promptable=True,
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    mask=input_mask,
                    words=input_word,
                )

                image_hr = image_hr.transpose(1, 2, 0) 
                mask_hr = mask_hr.transpose(1, 2, 0).astype(np.uint8)
                # if prompt_type == 'mask_all':
                #     mask_hr = find_overlapping_regions(mask_hr, input_mask[:,:,np.newaxis]/255)
                #     if len(mask_hr)!=0:
                #         mask_hr = mask_hr[0]
                #         mask_hr = mask_hr['mask']
                #         mask_hr = mask_hr[:,:,np.newaxis].astype(np.uint8)
                #     else:
                #         mask_hr = input_mask[:,:,np.newaxis]/255
                #     # print(gt_mask.shape,type(mask_hr))
                #     # print(mask_hr.shape,type(mask_hr),np.max(mask_hr),np.min(mask_hr))
                # # mask_hr = input_mask[:,:,np.newaxis]/255
                mask_blurred = cv2.GaussianBlur(mask_hr*255, (23, 23), 0)/255
                mask_blurred = mask_blurred[:,:,np.newaxis]
                mask_np = 1-(1-mask_hr) * (1-mask_blurred)
                com_image = (mask_hr * image_hr + (1-mask_hr) * image).astype(np.uint8)
                if flag:
                    blurred_com_image = image
                    pre_mask = mask_hr
                    flag = False
                # blurred_com_image = (mask_np * image_hr + (1-mask_np) * image).astype(np.uint8)
                blurred_com_image = (mask_np * image_hr + (1-mask_np) * blurred_com_image).astype(np.uint8)
                pre_mask = np.clip(pre_mask + mask_hr,0,1)
            io.imsave(args.output + 'image/' + basename +'.png', blurred_com_image)
            # cv2.imwrite(args.output + 'mask/' + basename + '.png', (mask_hr*255).astype(np.uint8))
            # cv2.imwrite(args.output + 'mask/' + basename + '.png', (mask_hr*255).astype(np.uint8))
            cv2.imwrite(args.output + 'mask/' + basename + '.png', (pre_mask*255).astype(np.uint8))
            # io.imsave('ori_image.png', image)

            gt = label * gt_mask[:,:,np.newaxis] + image * (1-gt_mask[:,:,np.newaxis])
            # gt = label * mask_np + image * (1-mask_np)
            io.imsave(args.output + 'label/' + basename + '.png', gt.astype(np.uint8))
            mse = torch.mean((torch.from_numpy(blurred_com_image) - torch.from_numpy(gt)) ** 2)
            PIXEL_MAX = 255.0
            psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
            psnr_sum += psnr.item()
            psnr_num = psnr_num + 1
            mae_sum += torch.mean(torch.abs(torch.from_numpy(blurred_com_image) - torch.from_numpy(gt)))
            # SSIM_sum += ssim(blurred_com_image, gt.astype(np.uint8), channel_axis=2, data_range=(255, 255))
            # print('PSNR: ', psnr.item())
            # exit(0)

            # mse2 = torch.mean((torch.from_numpy(com_image) - torch.from_numpy(gt)) ** 2)
            # psnr2 = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse2))
            # psnr_sum2 += psnr2.item()
            # psnr_num2 = psnr_num2 + 1
            # print('PSNR2: ', psnr2.item())
            # exit(0)
        else:
            raise RuntimeError("--promptable and --erase_mode must be set.")
    print('PSNR: ', psnr_sum / psnr_num)
    # print('PSNR: ', psnr_sum2 / psnr_num2)
    # print('SSIM: ', SSIM_sum / psnr_num)
    print('L1: ', mae_sum / psnr_num)

