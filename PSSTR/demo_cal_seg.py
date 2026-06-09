import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def calculate_iou(predicted, ground_truth):
    intersection = np.sum(np.logical_and(predicted, ground_truth))
    union = np.sum(np.logical_or(predicted, ground_truth))
    iou = intersection / union if union != 0 else 0.0
    return iou

def calculate_f1(predicted, ground_truth):
    # TP = np.logical_and(predicted, ground_truth).sum()
    # FP = np.logical_and(predicted, np.logical_not(ground_truth)).sum()
    # FN = np.logical_and(np.logical_not(predicted), ground_truth).sum()
    
    # precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    # recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    
    # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    # return f1数
    tp = np.sum(np.logical_and(predicted, ground_truth))
    fp = np.sum(np.logical_and(predicted, np.logical_not(ground_truth)))
    fn = np.sum(np.logical_and(np.logical_not(predicted), ground_truth))
    # print(tp,fp,fn)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

if __name__=='__main__':
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask\\' # 0.6657
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask_lora_mask\\' # 0.7188 
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask_nolora_mask\\' # 0.6619
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask_lora_wordmask\\' # 0.7208
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask199prwordmask\\' # 0.7208
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask050prwordmask\\' # 0.6657
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask050prmask\\' # 0.6619
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\word\\mask199prmask\\' # 0.7188
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_result\\word_all\\mask'  # 0.7319 0.8281
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_result\\mask_all\\mask'
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\box\\mask199prmask\\' # 0.8757 0.9319
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\box\\mask199prwordmask\\' # 0.8771 0.9326
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\box\\mask050prmask\\' # 0.4031 0.5352
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\box\\mask050prwordmask\\' # 0.4064 0.5352
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_result\\box_all\\mask'  # 0.8571 0.9212
    # gt_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\box\\gt\\'
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\point\\mask199prmask\\' # 0.8570 0.9184
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\point\\mask199prwordmask\\' # 0.8586 0.9193
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\point\\mask050prmask\\' # 0.3820 0.5114
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\point\\mask050prwordmask\\' # 0.3856 0.5120
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_result\\point_all\\mask'  # 0.8437 0.9115
    # gt_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\point\\gt\\'
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\mask\\mask199prmask\\' # 0.5107 0.6452
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\mask\\mask199prwordmask\\' # 0.5133 0.6456
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\mask\\mask050prmask\\' # 0.2367 0.3297
    # pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\mask\\mask050prwordmask\\' # 0.2424 0.3345
    pre_path = 'C:\\Users\\lg\\Hi-SAM\\output_result\\mask_all\\mask'  # 0.8443 0.9122
    gt_path = 'C:\\Users\\lg\\Hi-SAM\\output_seg_result\\mask\\gt\\'
    pre_list = [os.path.join(pre_path, fname) for fname in os.listdir(pre_path)]
    IoU_sum = 0.0
    F1_score_sum = 0.0
    count = 0
    for path in tqdm(pre_list):
        img = np.array(Image.open(path))
        img = (img > 10).astype(np.uint8)
        fname = path.split('\\')[-1].split('.')[0]
        # print(fname)
        gt_mask = os.path.join(gt_path, fname + '.png')
        # print(gt_mask)
        gt = np.array(Image.open(gt_mask))
        gt = (gt > 10).astype(np.uint8)

        # print(np.max(img), np.min(img))
        IoU = calculate_iou(img, gt)
        F1_score = calculate_f1(img, gt)
        IoU_sum += IoU
        F1_score_sum += F1_score
        # print(IoU,F1_score)
        # exit()
        count += 1
    print(f'IoU:{IoU_sum/count}')
    print(f'F1:{F1_score_sum/count}')