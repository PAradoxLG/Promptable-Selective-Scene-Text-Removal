import os
import argparse
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
import time
import datetime

from unistr import UniStR
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.loss import loss_masks, loss_hi_masks, loss_iou_mse, loss_hi_iou_mse, loss_image, percetual_loss, style_loss
from hi_sam.data.dataloader import get_im_gt_name_dict, create_dataloaders, train_transforms, eval_transforms, custom_collate_fn
from hi_sam.evaluation import Evaluator
from dinet.model_util import create_decoder
from dinet.vgg16 import VGG16
import utils.misc as misc
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--output", type=str, default="work_dirs/", 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default="./pretrained_checkpoint/sam_tss_l_hiertext.pth",
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument("--train_datasets", type=str, nargs='+', default=['flickrst_train'])
    parser.add_argument("--val_datasets", type=str, nargs='+', default=['flickrst_test'])
    parser.add_argument("--hier_det", action='store_true',
                        help="If False, only text stroke segmentation.")
    
    parser.add_argument("--promptable", default=True, action='store_true',
                        help="If False, only text stroke segmentation.")
    parser.add_argument("--unimask_decoder_weight", type=str,
                        help="pretrained unimask decoder.")
    parser.add_argument("--word_prompt", action='store_true',
                        help="If False, not support word prompt for segmentation")
    parser.add_argument("--word_embedding_weight", type=str, default="./pretrained_checkpoint/text_encoder.pth",
                        help="pretrained word decoder.")
    

    parser.add_argument("--erase_mode", action='store_true',
                        help="If False, only segment text for prompts")
    parser.add_argument('--np_per_image', default=2, type=int,
                        help='prompt num per image for text erasing.')
    # parser.add_argument("--erase_decoder_weight", type=str, default=r'work_dirs\2025-04-03__203553\best.pth',
    #                     help="The path to erasde_decoder weight.")
    parser.add_argument("--erase_decoder_weight", type=str,
                        help="The path to erasde_decoder weight.")
    
    parser.add_argument("--vgg_pretrained_path", type=str, default=r'pretrained_checkpoint/vgg16-397923af.pth',
                        help="pretrained vgg checkpoint path.")

    # >>>>>>>>>>>>>>>>>> erase decoder params >>>>>>>>>>>>>>>>>>>>>>>
    parser.add_argument('--decoder', type=str, default='dualhelix')
    parser.add_argument('--Block', type=str, default='cnn')
    parser.add_argument('--InterBlock', type=str, default='cnn')
    # parser.add_argument('--SegAux', type=str, default='sim_ffn')
    parser.add_argument('--SegAux', type=str, default='gate_correct')
    parser.add_argument('--SegAuxList', nargs='+')
    # parser.add_argument('--RMAux', type=str, default='sim_ffn')
    parser.add_argument('--RMAux', type=str, default='gate_correct')
    parser.add_argument('--RMAuxList', nargs='+')
    parser.add_argument('--dec_num_block', nargs='+',type=int, default=[1, 2, 3])
    parser.add_argument('--aux_num_blocks', nargs='+',type=int, default=[2, 3, 3])
    parser.add_argument('--channels', nargs='+',type=int, default=[64, 128, 256])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_charac_mask_decoder_name', default=["mask_decoder"], type=str, nargs='+')
    parser.add_argument('--lr_charac_mask_decoder', default=1e-5, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=150, type=int)
    parser.add_argument('--max_epoch_num', default=600, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--valid_period', default=3, type=int)
    parser.add_argument('--model_save_fre', default=100, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')

    return parser.parse_args()


def main(train_datasets, valid_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_datasets_names = [train_ds["name"] for train_ds in train_datasets]
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(
            train_im_gt_list,
            my_transforms=train_transforms,
            batch_size=args.batch_size_train,
            training=True,
            promptable=args.promptable,
            collate_fn=custom_collate_fn
        )
        print('type(train_dataloaders)',type(train_dataloaders))
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_datasets_names = [val_ds["name"] for val_ds in valid_datasets]
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_im_gt_list,
        my_transforms=eval_transforms,
        batch_size=args.batch_size_valid,
        training=True,
        promptable=args.promptable,
        collate_fn=custom_collate_fn
    )
    # valid_dataloaders, valid_datasets = create_dataloaders(
    #     valid_im_gt_list,
    #     my_transforms=eval_transforms,
    #     batch_size=args.batch_size_valid,
    #     promptable=args.promptable,
    #     training=False
    # )
    # print('type(valid_dataloaders)',type(valid_dataloaders))
    print(len(valid_dataloaders), " valid dataloaders created")
    # a=0
    # assert a==1

    ### --- Step 2: DistributedDataParallel---
    
    sam = model_registry[args.model_type](args=args)
    erase_decoder = create_decoder(args.decoder, args.Block, args.InterBlock,
                                   args.SegAux, args.RMAux,
                                   args.dec_num_block, args.aux_num_blocks,
                                   args.channels)
    
    # lg20250310
    target_lora_model = ['image_encoder.blocks.{}.Space_Adapter.D_fc1'.format(i) for i in range(sam.image_encoder.depth)] + ['image_encoder.blocks.{}.Space_Adapter.D_fc2'.format(i) for i in range(sam.image_encoder.depth)] + ['image_encoder.blocks.{}.MLP_Adapter.D_fc1'.format(i) for i in range(sam.image_encoder.depth)] + ['image_encoder.blocks.{}.MLP_Adapter.D_fc2'.format(i) for i in range(sam.image_encoder.depth)]
    lora_config = LoraConfig(target_modules=target_lora_model, r=8)
    lora_sam = get_peft_model(sam, lora_config)
    if args.unimask_decoder_weight:
        unimask_decoder_path = args.unimask_decoder_weight
        with open(unimask_decoder_path, "rb") as f:
            param_dict = torch.load(f)
        dict_keys = param_dict.keys()
        if 'optimizer' in dict_keys or 'lr_scheduler' in dict_keys or 'epoch' in dict_keys:
            model_param_dict = param_dict['model']
            if 'image_encoder' in model_param_dict.keys():
                # image_encoder_dict = param_dict['image_encoder']
                # info = lora_sam.image_encoder.load_state_dict(image_encoder_dict, strict=False)
                info = lora_sam.image_encoder.load_state_dict(model_param_dict['image_encoder'], strict=False)
                print(f'image_encoder matched info: {info}')
        del param_dict
        del model_param_dict
        # if not args.erase_mode:
        #     for p in lora_sam.unimask_decoder.parameters():
        #         p.requires_grad = True
        for p in lora_sam.unimask_decoder.parameters():
            p.requires_grad = True
    # if args.erase_mode:
    #     for p in lora_sam.parameters():
    #         p.requires_grad = False

    # model = UniStR(sam, erase_decoder)

    if args.erase_decoder_weight:
        with open(args.erase_decoder_weight, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        dict_keys = state_dict.keys()
        if 'optimizer' in dict_keys or 'lr_scheduler' in dict_keys or 'epoch' in dict_keys:
            param2_dict = state_dict['model']
        del state_dict
        info1 = erase_decoder.load_state_dict(param2_dict['erase_decoder'], strict=False)
        print(f'erase_model matched info: {info1}')
        info2 = lora_sam.image_encoder.load_state_dict(param2_dict['image_encoder'], strict=False)
        print(f'image_encoder matched info: {info2}')
        info3 = lora_sam.unimask_decoder.load_state_dict(param2_dict['unimask_decoder'], strict=False)
        print(f'unimask_decoder matched info: {info3}')
    else:
        for p in lora_sam.parameters():
            p.requires_grad = False

    model = UniStR(lora_sam, erase_decoder)

    # if torch.cuda.is_available():
    #     model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    # model_without_ddp = model.module
 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        n_parameters = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ('unimask_decoder' in n or 'word_embedding' in n or 'image_encoder' in n or 'erase_decoder' in n))
        print(f'number of trainable params: {n_parameters/1e6:.4f}M')
        # for n, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(n)

        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        param_dicts = [
            {
                # "params": [p for n, p in model_without_ddp.named_parameters()
                #            if not match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad],
                "params": [p for n, p in model.named_parameters() if not match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad],
                "lr": args.lr
            },
            {
                # "params": [p for n, p in model_without_ddp.named_parameters()
                #            if match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad],
                "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad],
                "lr": args.lr_charac_mask_decoder
            }
        ]
        # print('$#'*100,[n for n, p in model.named_parameters() if not match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad])
        # print('$#'*100,[n for n, p in model.named_parameters() if match_name_keywords(n, args.lr_charac_mask_decoder_name) and p.requires_grad])
        if not args.erase_mode:
            optimizer = optim.AdamW(param_dicts, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)
        else:
            print('----------------erase_optimizer---------------')
            optimizer = optim.AdamW(param_dicts, lr=args.lr, betas=(0.9, 0.999))
        # optimizer = optim.AdamW(param_dicts, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch
        train(args, model, optimizer, train_dataloaders, train_datasets_names, lr_scheduler, valid_dataloaders, valid_datasets_names)
    else:
        print("restore model from:", args.checkpoint)
        evaluate(args, model, valid_dataloaders, valid_datasets_names)


def train(args, model, optimizer, train_dataloaders, train_datasets_names, lr_scheduler, valid_dataloaders, valid_datasets_names):
    # valid_dataloaders = valid_dataloaders[0]
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)
    best_iou = [-1 for _ in range(len(valid_datasets_names))]
    best_psnr = 0

    start_time = datetime.datetime.now().strftime('%Y-%m-%d__%H%M%S')
    model.train()
    _ = model.to(device=args.device)
    # print('$#'*100,model.device)
    from torch.cuda.amp import autocast, GradScaler
    gradsclaler = GradScaler()

    vgg16 = VGG16(args.vgg_pretrained_path).to(model.device)

    for epoch in range(epoch_start, epoch_num):
        print("epoch: ", epoch, " lr: ", optimizer.param_groups[0]["lr"])
        # print("epoch: ", epoch, " lr: ", optimizer.param_groups[1]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        # train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        
        for data in metric_logger.log_every(train_dataloaders, 50):
            inputs, labels = data['image'].to(model.device), data['label'].to(model.device)  # (bs,3,1024,1024), (bs,1,1024,1024)
            no_text_images = data['no_text_images']
            
            # print("$$$$$$$$$$$$$$$")
            # for i, no_text_image in enumerate(no_text_images):
            #     print(f"no_text_image[{i}].shape <<<<<< ", no_text_image.shape)
            
            batched_input = []
            # if args.hier_det:
            #     para_masks, line_masks, word_masks = data['paragraph_masks'], data['line_masks'], data['word_masks']
            #     line2para_idx = data['line2paragraph_index']
            #     fg_points, para_masks, line_masks, word_masks = misc.sample_foreground_points(labels, para_masks, line_masks, word_masks, line2para_idx)

            # lg20250305
            # input_keys = ['box','point', 'mask']
            input_keys = ['box','point', 'mask', 'word']
            # input_keys = ['word']
            # input_keys = ['mask','word']
            if args.promptable:
                word_masks = data['word_masks']
                input_type = random.choice(input_keys)

                fg_prompt = None
                if input_type == 'point':
                    fg_points, word_masks, no_text_images = misc.sample_points(labels, word_masks, inputs, no_text_images)
                    fg_prompt = fg_points
                elif input_type == 'box':
                    fg_boxes, word_masks, no_text_images = misc.masks_to_boxes(labels, word_masks, inputs, no_text_images)
                    fg_prompt = fg_boxes
                elif input_type == 'mask':
                    fg_mask_points, word_masks, no_text_images = misc.masks_to_points(labels, word_masks, inputs, no_text_images)
                    fg_prompt = fg_mask_points
                elif input_type == 'word':
                    word_list = data['word_list']

                # print('#$'*10, [mask.shape[0] for mask in word_masks], len(word_masks), word_list, len(word_list))

                # 擦除模式，默认分割已取得较好效果，对输入规模进行裁剪，仅用于训练erase_decoder（由于重建所需较大的显存，裁剪以缓解需求）
                if args.erase_mode:
                    if fg_prompt != None:
                        min_per_image = args.np_per_image

                        prompt_nums = torch.tensor([x.shape[0] for x in fg_prompt]) # [10, 9, 8, 2]
                        prompt_tot = torch.cumsum(prompt_nums, dim=0) # [10, 19, 27, 29]
                        start_idx = prompt_tot - prompt_nums # [0, 10, 19, 27]
                        new_word_masks = [word_masks[start:end] for start, end in zip(start_idx.tolist(), prompt_tot.tolist())]
                                        
                        min_per_image = min(min_per_image, prompt_nums.min().item())
                        pos_idx = torch.tensor([
                            random.sample(range(0, prompt_num), min_per_image) for prompt_num in prompt_nums
                        ])
                        new_word_masks = [word_mask[pos] for word_mask, pos in zip(new_word_masks, pos_idx)]
                        fg_prompt = [prompt[pos] for prompt, pos in zip(fg_prompt, pos_idx)]

                        if input_type == 'point':
                            fg_points = fg_prompt
                        elif input_type == 'box':
                            fg_boxes = fg_prompt
                        elif input_type == 'mask':
                            fg_mask_points = fg_prompt
                        
                        word_masks = torch.cat(new_word_masks,dim=0)
                        no_text_images = [no_text_image[pos] for no_text_image, pos in zip(no_text_images, pos_idx)]
                        # for idx, t in enumerate(no_text_images):
                        #     print(f't[{idx}].shape ', t.shape)
                        no_text_images = torch.cat(no_text_images)
                        # print(word_masks.shape)
                    else:
                        min_per_image = args.np_per_image

                        mask_nums = torch.tensor([mask.shape[0] for mask in word_masks])
                                        
                        min_per_image = min(min_per_image, mask_nums.min().item())
                        pos_idx = torch.tensor([
                            random.sample(range(0, mask_num), min_per_image) for mask_num in mask_nums
                        ])
                        # pos_idx = torch.tensor([
                        #     random.sample(range(0, mask_num), min(min_per_image, mask_num)) for mask_num in mask_nums
                        # ])
                        new_word_masks = [word_mask[pos] for word_mask, pos in zip(word_masks, pos_idx)]
                        word_list = [[words[i] for i in pos] for words, pos in zip(word_list, pos_idx)]
                        # print('@#'*10, [mask.shape[0] for mask in new_word_masks], new_word_list)

                        # word_masks = torch.cat(new_word_masks,dim=0)
                        word_masks = torch.cat(new_word_masks,dim=0).unsqueeze(1).to(model.device)
                        no_text_images = [no_text_image[pos] for no_text_image, pos in zip(no_text_images, pos_idx)]
                        no_text_images = torch.cat(no_text_images).to(model.device)
                        
                        # print(word_masks.shape)
                        # assert fg_prompt != None

            # print("inputs.shape >>>>>>>>> ", inputs.shape)
            # print("labels.shape >>>>>>>>> ", labels.shape)
            # print("word_masks.shape >>>>> ", word_masks.shape)
            # for i, no_text_image in enumerate(no_text_images):
            #     print(f"no_text_image[{i}].shape > ", no_text_image.shape)
            # print(">>>>>>>>>> loaded data >>>>>>>>>>>>>>>>")

            for b_i in range(len(inputs)):
                dict_input = dict()
                dict_input['image'] = inputs[b_i].to(model.device).contiguous()
                dict_input['original_size'] = inputs[b_i].shape[-2:]

                if args.promptable:
                    if input_type == 'box':
                        dict_input['boxes'] = fg_boxes[b_i]
                    elif input_type == 'point':
                        point_coords = fg_points[b_i][:, None, :] # Every Image: (words * num, 1, 2)
                        dict_input['point_coords'] = point_coords
                        dict_input['point_labels'] = torch.ones((point_coords.shape[0], point_coords.shape[1]), device=point_coords.device)
                    elif input_type == 'mask':
                        point_coords = fg_mask_points[b_i] # Every Image: (words, num, 2)
                        dict_input['point_coords'] = point_coords
                        dict_input['point_labels'] = torch.ones((point_coords.shape[0], point_coords.shape[1]), device=point_coords.device)
                    elif input_type == 'word':
                        dict_input['word'] = word_list[b_i]
                        # word_masks = [x.unsqueeze(1).to(labels.device) for x in word_masks]
                        # word_masks = torch.cat(word_masks, dim=0)


                batched_input.append(dict_input)

            with autocast():
                if args.promptable:
                    
                    if not args.erase_mode:
                        pr_masks_logits, pr_iou_output, pr_word_masks_logits = model(batched_input)
                        # loss_focal, loss_dice = loss_masks(up_masks_logits, labels / 255.0, len(up_masks_logits))
                        # loss_mse = loss_iou_mse(iou_output, up_masks, labels)
                        # loss_lr = loss_focal * 20 + loss_dice + loss_mse


                        if word_masks is not None:
                            loss_focal_word, loss_dice_word = loss_hi_masks(
                                pr_masks_logits[:, 0:1, :, :], word_masks, len(pr_masks_logits)
                            )
                            loss_focal_word_384, loss_dice_word_384 = loss_hi_masks(
                                pr_word_masks_logits[:, 0:1, :, :], word_masks, len(pr_word_masks_logits),
                            )

                            loss_mse_word = loss_hi_iou_mse(
                                pr_iou_output[:, 0:1], pr_masks_logits[:, 0:1, :, :], model.mask_threshold, word_masks
                            )

                            # loss = loss_lr + loss_hr + loss_word + loss_word_384 + loss_line + loss_para * 0.5
                            loss = loss_focal_word + loss_dice_word + loss_focal_word_384  + loss_dice_word_384 + loss_mse_word
                            loss_dict = {
                                "loss_focal_word": loss_focal_word,
                                "loss_dice_word": loss_dice_word,
                                "loss_focal_word_384": loss_focal_word_384,
                                "loss_dice_word_384": loss_dice_word_384,
                                "loss_mse_word": loss_mse_word
                            }
                    else:
                        pr_masks_logits, pr_iou_output, pr_word_masks_logits, outputs, masks = model(batched_input)
                        # sizes = [64, 128, 256, 1024]
                        masks_by_size = [torch.cat([mask[i] for mask in masks]) for i in range(4)]
                        masks_64, masks_128, masks_256, masks_1024 = masks_by_size

                        # print('-'*100)
                        # print('batched_input',len(batched_input))
                        # print('masks_64.shape',masks_64.shape)
                        # print('masks_128.shape',masks_128.shape)
                        # print('masks_256.shape',masks_256.shape)
                        # print('masks_1024.shape',masks_1024.shape)

                        word_masks_64 = torch.clip(F.interpolate(word_masks, masks_64.shape[-2:], mode='bilinear'), 0.0, 1.0)
                        word_masks_128 = torch.clip(F.interpolate(word_masks, masks_128.shape[-2:], mode='bilinear'), 0.0, 1.0)
                        word_masks_256 = torch.clip(F.interpolate(word_masks, masks_256.shape[-2:], mode='bilinear'), 0.0, 1.0)

                        # print('word_masks_64.shape',word_masks_64.shape)
                        # print('word_masks_128.shape',word_masks_128.shape)
                        # print('word_masks_256.shape',word_masks_256.shape)
                        # print('word_masks_1024.shape',word_masks.shape)
                        # print(masks_64.device, word_masks_64.device)

                        loss_focal_64, loss_dice_64 = loss_masks(masks_64, word_masks_64, len(word_masks))
                        loss_focal_128, loss_dice_128 = loss_masks(masks_128, word_masks_128, len(word_masks))
                        loss_focal_256, loss_dice_256 = loss_masks(masks_256, word_masks_256, len(word_masks))
                        loss_focal_1024, loss_dice_1024 = loss_masks(masks_1024, word_masks, len(word_masks))
                        
                        outputs_by_size = [torch.cat([output[i] for output in outputs]) for i in range(4)]
                        outputs_64, outputs_128, outputs_256, outputs_1024 = outputs_by_size
                        # print(outputs_64.shape)
                        # print(outputs_128.shape)
                        # print(outputs_256.shape)
                        # print(outputs_1024.shape) # [b*n,3,1024,1024]
                        
                        no_text_images = no_text_images / 255.0
                        # print('no_text_images min and max',no_text_images.min(), no_text_images.max()) # 0-1
                        # print('outputs_1024 min and max',outputs_1024.min(), outputs_1024.max()) # -0.0594-1.0898
                        # nt_image = no_text_images[0] # 0-1
                        # ori_image = inputs[0] # 0-255
                        # mask_image = word_masks[0] # 0-1
                        # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'nt_image'+'.png', (nt_image*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                        # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'input'+'.png', (ori_image).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                        # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'word_mask'+'.png', (mask_image*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                        no_text_images_64 = torch.clip(F.interpolate(no_text_images, outputs_64.shape[-2:], mode='bilinear'), 0.0, 1.0)
                        no_text_images_128 = torch.clip(F.interpolate(no_text_images, outputs_128.shape[-2:], mode='bilinear'), 0.0, 1.0)
                        no_text_images_256 = torch.clip(F.interpolate(no_text_images, outputs_256.shape[-2:], mode='bilinear'), 0.0, 1.0)

                        l1loss_64 = loss_image(outputs_64, no_text_images_64)
                        l1loss_128 = loss_image(outputs_128, no_text_images_128)
                        l1loss_256 = loss_image(outputs_256, no_text_images_256)
                        l1loss_1024 = loss_image(outputs_1024, no_text_images)

                        # lg add 20250330
                        # print('word_masks.shape',word_masks.shape) #[2, 1, 1024, 1024]
                        # print('inputs.shape',inputs.shape) #[2, 3, 1024, 1024]
                        # outputs_com_1024 = outputs_1024 * word_masks + (1 - word_masks) * inputs * 0.1
                        eps = 1e-5
                        mask_total_1024 = torch.tensor([word_masks.shape[2]*word_masks.shape[3]]).to(model.device)
                        mask_total_256 = torch.tensor([word_masks_256.shape[2]*word_masks.shape[3]]).to(model.device)
                        mask_total_128 = torch.tensor([word_masks_128.shape[2]*word_masks.shape[3]]).to(model.device)
                        mask_total_64 = torch.tensor([word_masks_64.shape[2]*word_masks.shape[3]]).to(model.device)
                        mask_sum_1024 = torch.sum(word_masks, dim=(1,2,3))
                        mask_sum_256 = torch.sum(word_masks_256, dim=(1,2,3))
                        mask_sum_128 = torch.sum(word_masks_128, dim=(1,2,3))
                        mask_sum_64 = torch.sum(word_masks_64, dim=(1,2,3))
                        mask_scale_1024 = mask_total_1024/(mask_sum_1024 + eps)
                        mask_scale_256 = mask_total_256/(mask_sum_256 + eps)
                        mask_scale_128 = mask_total_128/(mask_sum_128 + eps)
                        mask_scale_64 = mask_total_64/(mask_sum_64 + eps)
                        mask_scale_1024 = mask_scale_1024.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(model.device)
                        mask_scale_256 = mask_scale_256.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(model.device)
                        mask_scale_128 = mask_scale_128.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(model.device)
                        mask_scale_64 = mask_scale_64.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(model.device)

                        outputs_mask_1024 = outputs_1024 * word_masks * mask_scale_1024
                        # inputs_256 = torch.clip(F.interpolate(no_text_images, outputs_256.shape[-2:], mode='bilinear'), 0.0, 1.0)
                        # outputs_com_256 = outputs_256 * word_masks_256 + (1 - word_masks_256) * inputs_256 * 0.1
                        outputs_mask_256 = outputs_256 * word_masks_256 * mask_scale_256
                        outputs_mask_128 = outputs_128 * word_masks_128 * mask_scale_128
                        outputs_mask_64 = outputs_64 * word_masks_64 * mask_scale_64

                        l1loss_mask_1024 = loss_image(outputs_mask_1024, no_text_images * word_masks * mask_scale_1024)
                        l1loss_mask_256 = loss_image(outputs_mask_256, no_text_images_256 * word_masks_256 * mask_scale_256)
                        l1loss_mask_128 = loss_image(outputs_mask_128, no_text_images_128 * word_masks_128 * mask_scale_128)
                        l1loss_mask_64 = loss_image(outputs_mask_64, no_text_images_64 * word_masks_64 * mask_scale_64)
                        # end

                        # vgg16 feats >>>>>>>>>>>>>>>>>>>>>
                        feat_outp_1024 = vgg16(outputs_1024)
                        # feat_com_1024 = vgg16(outputs_com_1024)
                        feat_gt_1024 = vgg16(no_text_images)
                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                        pcrloss1024 = percetual_loss(feat_outp_1024, feat_gt_1024)
                        # pcrloss_com1024 = percetual_loss(feat_com_1024, feat_gt_1024)

                        styleloss1024 = style_loss(feat_outp_1024, feat_gt_1024)
                        # styleloss_com1024 = style_loss(feat_com_1024, feat_gt_1024)

                        feat_outp_256 = vgg16(outputs_256)
                        feat_gt_256 = vgg16(no_text_images_256)
                        pcrloss256 = percetual_loss(feat_outp_256, feat_gt_256)
                        styleloss256 = style_loss(feat_outp_256, feat_gt_256)

                        feat_outp_128 = vgg16(outputs_128)
                        feat_gt_128 = vgg16(no_text_images_128)
                        pcrloss128 = percetual_loss(feat_outp_128, feat_gt_128)

                        feat_outp_64 = vgg16(outputs_64)
                        feat_gt_64 = vgg16(no_text_images_64)
                        pcrloss64 = percetual_loss(feat_outp_64, feat_gt_64)

                        if word_masks is not None:
                            loss_focal_word, loss_dice_word = loss_hi_masks(
                                pr_masks_logits[:, 0:1, :, :], word_masks, len(pr_masks_logits)
                            )
                            loss_focal_word_384, loss_dice_word_384 = loss_hi_masks(
                                pr_word_masks_logits[:, 0:1, :, :], word_masks, len(pr_word_masks_logits),
                            )

                            loss_mse_word = loss_hi_iou_mse(
                                pr_iou_output[:, 0:1], pr_masks_logits[:, 0:1, :, :], model.mask_threshold, word_masks
                            )

                            loss_seg_branch = loss_focal_word + loss_dice_word + loss_focal_word_384  + loss_dice_word_384 + loss_mse_word

                        loss = l1loss_64 * 5 + l1loss_128 * 5 + l1loss_256 * 5 + l1loss_1024 * 5 \
                               + loss_focal_64 + loss_dice_64 + loss_focal_128 + loss_dice_128 \
                               + loss_focal_256 + loss_dice_256 + loss_focal_1024 + loss_dice_1024 \
                               + pcrloss1024 * 5 + pcrloss256 * 1 + pcrloss128 * 1 + pcrloss64 * 1 + loss_seg_branch \
                               + l1loss_mask_64 * 1 + l1loss_mask_128 * 1 + l1loss_mask_256 * 2 + l1loss_mask_1024 * 5 \
                               + styleloss1024 * 250 + styleloss256 * 250
                        # loss = l1loss_64 * 10 + l1loss_128 * 10 + l1loss_256 * 10 + l1loss_1024 * 10 \
                        #        + loss_focal_64  + loss_dice_64  + loss_focal_128  + loss_dice_128  \
                        #        + loss_focal_256  + loss_dice_256  + loss_focal_1024  + loss_dice_1024  \
                        #        + pcrloss1024 + pcrloss256 + pcrloss128 \
                        loss_dict = {
                                "l1_mask_1024": l1loss_mask_1024 * 5,
                                "l1_1024": l1loss_1024 * 10,
                                "pcrloss_1024": pcrloss1024 * 5,
                                # "pcrloss_com1024": pcrloss_com1024 * 0.01,
                                "styleloss1024": styleloss1024 * 250,
                                # "styleloss_com1024": styleloss_com1024 * 120,
                                "loss_dice_1024": loss_dice_1024,
                                "loss_focal_1024": loss_focal_1024,
                                # "l1_com_256": l1loss_com_256 * 10,
                                'l1_mask_256': l1loss_mask_256 * 2,
                                "l1_256": l1loss_256 * 10,
                                "pcrloss256": pcrloss256 * 1,
                                "styleloss256": styleloss256 * 250,
                                "loss_dice_256": loss_dice_256,
                                "loss_focal_256": loss_focal_256,
                                'l1_mask_128': l1loss_mask_128 * 1,
                                "l1_128": l1loss_128 * 10,
                                "pcrloss128": pcrloss128 * 1,
                                "loss_dice_128": loss_dice_128,
                                "loss_focal_128": loss_focal_128,
                                'l1_mask_64': l1loss_mask_64 * 1,
                                "l1_64": l1loss_64 * 10,
                                "pcrloss64": pcrloss64 * 1,
                                "loss_dice_64": loss_dice_64,
                                "loss_focal_64": loss_focal_64,
                                "loss_focal_word": loss_focal_word,
                                "loss_dice_word": loss_dice_word,
                                "loss_focal_word_384": loss_focal_word_384,
                                "loss_dice_word_384": loss_dice_word_384,
                                "loss_mse_word": loss_mse_word
                            }
                        # loss_dict = {
                        #         # "l1_com_1024": l1loss_com_1024 * 10,
                        #         "l1_1024": l1loss_1024 * 10,
                        #         "pcrloss_1024": pcrloss1024,
                        #         # "pcrloss_com1024": pcrloss_com1024 * 0.01,
                        #         # "styleloss1024": styleloss1024 * 120,
                        #         # "styleloss_com1024": styleloss_com1024 * 120,
                        #         "loss_dice_1024": loss_dice_1024 ,
                        #         "loss_focal_1024": loss_focal_1024 ,
                        #         # "l1_com_256": l1loss_com_256 * 10,
                        #         "l1_256": l1loss_256 * 10,
                        #         "pcrloss256": pcrloss256,
                        #         # "styleloss256": styleloss256 * 120,
                        #         "loss_dice_256": loss_dice_256 ,
                        #         "loss_focal_256": loss_focal_256 ,
                        #         "l1_128": l1loss_128 * 10,
                        #         "pcrloss128": pcrloss128,
                        #         "loss_dice_128": loss_dice_128 ,
                        #         "loss_focal_128": loss_focal_128 ,
                        #         "l1_64": l1loss_64 * 10,
                        #         "loss_dice_64": loss_dice_64 ,
                        #         "loss_focal_64": loss_focal_64 ,
                        #     }
                else:
                    raise RuntimeError("--promptable must be set.")
                # else:
                #     up_masks_logits, up_masks, iou_output, hr_masks_logits, hr_masks, hr_iou_output = model(
                #         batched_input, multimask_output=False
                #     )
                #     loss_focal, loss_dice = loss_masks(up_masks_logits, labels / 255.0, len(up_masks_logits))
                #     loss_focal_hr, loss_dice_hr = loss_masks(hr_masks_logits, labels / 255.0, len(up_masks_logits))
                #     loss_mse = loss_iou_mse(iou_output, up_masks, labels)
                #     loss_mse_hr = loss_iou_mse(hr_iou_output, hr_masks, labels)
                #     loss = loss_focal * 20 + loss_dice + loss_mse + loss_focal_hr * 20 + loss_dice_hr + loss_mse_hr
                #     loss_dict = {
                #         "loss_iou_mse": loss_mse,
                #         "loss_dice": loss_dice,
                #         "loss_focal": loss_focal * 20,
                #         "loss_iou_mse_hr": loss_mse_hr,
                #         "loss_dice_hr": loss_dice_hr,
                #         "loss_focal_hr": loss_focal_hr * 20,
                #     }
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                losses_reduced_scaled = sum(loss_dict_reduced.values())
                loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            gradsclaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            gradsclaler.step(optimizer)
            gradsclaler.update()
            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
        
        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        if (epoch - epoch_start) % args.valid_period == 0 or (epoch + 1) == epoch_num:
            # if args.hier_det:
            #     model.module.hier_det = False  # disable hi_decoder temporally
            # if args.promptable:
            #     model.promptable = False
            # test_stats = evaluate(args, model, valid_dataloaders, valid_datasets_names)
            # if args.promptable:
            #     model.hier_det = True
            checkpoint_dir = os.path.join(args.output, start_time)
            os.makedirs(checkpoint_dir, exist_ok=True)
            if misc.is_main_process():
                # lg20250305
                # save_dict = model.erase_decoder.state_dict() if args.erase_mode else model.mask_extractor.unimask_decoder.state_dict()
                if args.erase_mode:
                    # save_dict = model.erase_decoder.state_dict()
                    save_dict = {
                        'erase_decoder': model.erase_decoder.state_dict(),
                        'unimask_decoder': model.mask_extractor.unimask_decoder.state_dict(),
                        'image_encoder': model.mask_extractor.image_encoder.state_dict(),
                    }
                else:
                    save_dict = {
                        'unimask_decoder': model.mask_extractor.unimask_decoder.state_dict(),
                        'word_chan_proj': model.mask_extractor.word_chan_proj.state_dict(),
                        'image_encoder': model.mask_extractor.image_encoder.state_dict(),
                    }
                # end
                checkpoint = {
                    'model': save_dict,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }
                # torch.save(checkpoint, os.path.join(checkpoint_dir, f"{epoch:03d}.pth"))
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"last.pth"))
            # lg 20250330
            model.eval()
            iterate = -1
            psnr_list_sum = 0
            psnr_num_sum = 0

            # for idx, data in enumerate(valid_dataloaders):
            with torch.no_grad():
                for data in metric_logger.log_every(valid_dataloaders, 50):
                    # print(type(data))
                    # for key in data.keys():
                    #     print(key)
                    inputs, labels = data['image'].to(model.device), data['label'].to(model.device)  # (bs,3,1024,1024), (bs,1,1024,1024)
                    # inputs, labels = inputs.unsqueeze(0), labels.unsqueeze(0)
                    no_text_images = data['no_text_images']
                    iterate += 1
                    batched_input = []
                    input_keys = ['box','point', 'mask', 'word']
                    if args.promptable:
                        word_masks = data['word_masks']
                        # print(word_masks[0].shape,labels.shape)
                        input_type = random.choice(input_keys)

                        fg_prompt = None
                        if input_type == 'point':
                            fg_points, word_masks, no_text_images = misc.sample_points(labels, word_masks, inputs, no_text_images)
                            fg_prompt = fg_points
                        elif input_type == 'box':
                            fg_boxes, word_masks, no_text_images = misc.masks_to_boxes(labels, word_masks, inputs, no_text_images)
                            fg_prompt = fg_boxes
                        elif input_type == 'mask':
                            fg_mask_points, word_masks, no_text_images = misc.masks_to_points(labels, word_masks, inputs, no_text_images)
                            fg_prompt = fg_mask_points
                        elif input_type == 'word':
                            word_list = data['word_list']

                        # 擦除模式，默认分割已取得较好效果，对输入规模进行裁剪，仅用于训练erase_decoder（由于重建所需较大的显存，裁剪以缓解需求）
                        if args.erase_mode:
                            if fg_prompt != None:
                                min_per_image = 1

                                prompt_nums = torch.tensor([x.shape[0] for x in fg_prompt]) # [2]
                                # print('prompt_nums',prompt_nums)
                                prompt_tot = torch.cumsum(prompt_nums, dim=0) # [2]
                                # print('prompt_tot',prompt_tot)
                                start_idx = prompt_tot - prompt_nums # [0]
                                # print('start_idx',start_idx)
                                new_word_masks = [word_masks[start:end] for start, end in zip(start_idx.tolist(), prompt_tot.tolist())]
                                # print(len(new_word_masks)) # 1
                                # print(type(new_word_masks[0])) # <class 'torch.Tensor'>
                                # print(new_word_masks[0].shape) # torch.Size([2, 1, 480, 640])
                                                
                                min_per_image = min(min_per_image, prompt_nums.min().item())
                                # print(min_per_image) # 1
                                pos_idx = torch.tensor([
                                    random.sample(range(0, prompt_num), min_per_image) for prompt_num in prompt_nums
                                ])
                                # print('pos_idx',pos_idx) # pos_idx tensor([[0]])
                                new_word_masks = [word_mask[pos] for word_mask, pos in zip(new_word_masks, pos_idx)]
                                # new_word_masks = [word_mask[pos.item()] for word_mask, pos in zip(new_word_masks, pos_idx)]
                                # for word_mask, pos in zip(new_word_masks, pos_idx):
                                #     print(pos)
                                #     a = word_mask[0] # ok
                                #     b = word_mask[pos.item()] #ok
                                #     c = word_mask[pos] # error
                                fg_prompt = [prompt[pos] for prompt, pos in zip(fg_prompt, pos_idx)]
                                # fg_prompt = [prompt[pos.item()] for prompt, pos in zip(fg_prompt, pos_idx)]

                                if input_type == 'point':
                                    fg_points = fg_prompt
                                elif input_type == 'box':
                                    fg_boxes = fg_prompt
                                elif input_type == 'mask':
                                    fg_mask_points = fg_prompt
                                
                                word_masks = torch.cat(new_word_masks,dim=0)
                                no_text_images = [no_text_image[pos] for no_text_image, pos in zip(no_text_images, pos_idx)]
                                # no_text_images = [no_text_image[pos.item()] for no_text_image, pos in zip(no_text_images, pos_idx)]
                                no_text_images = torch.cat(no_text_images)
                            else:
                                min_per_image = 1

                                mask_nums = torch.tensor([mask.shape[0] for mask in word_masks])
                                                
                                min_per_image = min(min_per_image, mask_nums.min().item())
                                pos_idx = torch.tensor([
                                    random.sample(range(0, mask_num), min_per_image) for mask_num in mask_nums
                                ])
                                # pos_idx = torch.tensor([
                                #     random.sample(range(0, mask_num), min(min_per_image, mask_num)) for mask_num in mask_nums
                                # ])
                                new_word_masks = [word_mask[pos] for word_mask, pos in zip(word_masks, pos_idx)]
                                # new_word_masks = [word_mask[pos.item()] for word_mask, pos in zip(word_masks, pos_idx)]
                                word_list = [[words[i] for i in pos] for words, pos in zip(word_list, pos_idx)]
                                # print('@#'*10, [mask.shape[0] for mask in new_word_masks], new_word_list)

                                # word_masks = torch.cat(new_word_masks,dim=0)
                                word_masks = torch.cat(new_word_masks,dim=0).unsqueeze(1).to(model.device)
                                no_text_images = [no_text_image[pos] for no_text_image, pos in zip(no_text_images, pos_idx)]
                                # no_text_images = [no_text_image[pos.item()] for no_text_image, pos in zip(no_text_images, pos_idx)]
                                no_text_images = torch.cat(no_text_images).to(model.device)
                                
                                # print(word_masks.shape)
                                # assert fg_prompt != None

                    for b_i in range(len(inputs)):
                        dict_input = dict()
                        dict_input['image'] = inputs[b_i].to(model.device).contiguous()
                        dict_input['original_size'] = inputs[b_i].shape[-2:]

                        if args.promptable:
                            if input_type == 'box':
                                dict_input['boxes'] = fg_boxes[b_i]
                            elif input_type == 'point':
                                point_coords = fg_points[b_i][:, None, :] # Every Image: (words * num, 1, 2)
                                dict_input['point_coords'] = point_coords
                                dict_input['point_labels'] = torch.ones((point_coords.shape[0], point_coords.shape[1]), device=point_coords.device)
                            elif input_type == 'mask':
                                point_coords = fg_mask_points[b_i] # Every Image: (words, num, 2)
                                dict_input['point_coords'] = point_coords
                                dict_input['point_labels'] = torch.ones((point_coords.shape[0], point_coords.shape[1]), device=point_coords.device)
                            elif input_type == 'word':
                                dict_input['word'] = word_list[b_i]

                        batched_input.append(dict_input)

                    if args.promptable:
                        pr_masks_logits, pr_iou_output, pr_word_masks_logits, outputs, masks = model(batched_input)
                        # sizes = [64, 128, 256, 1024]

                        outputs_by_size = [torch.cat([output[i] for output in outputs]) for i in range(4)]
                        outputs_64, outputs_128, outputs_256, outputs_1024 = outputs_by_size
                        # print(outputs_64.shape)
                        # print(outputs_128.shape)
                        # print(outputs_256.shape)
                        # print(outputs_1024.shape) # [b*n,3,1024,1024]
                        
                        no_text_images = no_text_images / 255.0
                        # print('no_text_images min and max',no_text_images.min(), no_text_images.max()) # 0-1
                        # print('outputs_1024 min and max',outputs_1024.min(), outputs_1024.max()) # -0.0594-1.0898
                        nt_image = no_text_images[0] # 0-1
                        ori_image = inputs[0] # 0-255
                        # mask_image = word_masks[0] # 0-1
                        outputs_1024 = torch.clamp(outputs_1024, 0, 1) # 0-1
                        outputs_256 = torch.clamp(outputs_256, 0, 1) # 0-1
                        outputs_128 = torch.clamp(outputs_128, 0, 1) # 0-1
                        outputs_64 = torch.clamp(outputs_64, 0, 1) # 0-1
                        output = outputs_1024[0] 
                        # output256 = outputs_256[0] 
                        # output128 = outputs_128[0] 
                        output64 = outputs_64[0] 
                        # no_text_images_64 = torch.clip(F.interpolate(no_text_images, outputs_64.shape[-2:], mode='bilinear'), 0.0, 1.0)
                        # nt_image64 = no_text_images_64[0]

                        outputs_com_1024 = outputs_1024 * word_masks + (1 - word_masks) * (inputs/255)
                        output_com_1024 = outputs_com_1024[0]
                        
                        if iterate % 200 == 0:
                        # if idx % 50 == 0:
                            cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'nt_image_'+ str(epoch) + '_' + str(iterate) +'.png', (nt_image*255).flip(0).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'nt_image64_'+ str(epoch) + '_' + str(iterate) +'.png', (nt_image64*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'input_'+ str(epoch) + '_' + str(iterate)  +'.png', (ori_image).flip(0).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'word_mask_'+ str(epoch) + '_' + str(iterate)  +'.png', (mask_image*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'output_'+ str(epoch) + '_' + str(iterate)  +'.png', (output*255).flip(0).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'output256_'+ str(epoch) + '_' + str(iterate)  +'.png', (output256*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'output128_'+ str(epoch) + '_' + str(iterate)  +'.png', (output128*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            # cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'output64_'+ str(epoch) + '_' + str(iterate)  +'.png', (output64*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
                            cv2.imwrite('C:/Users/lg/Hi-SAM/log_vis/'+'outputcom_'+ str(epoch) + '_' + str(iterate)  +'.png', (output_com_1024*255).flip(0).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))

                        for i in range(word_masks.shape[0]):
                            # rec = outputs_com_1024[i]*255
                            rec = outputs_1024[i]*255 #使用组合的会有可能当word_mask为0时，rec与ori完全一致而导致psnr为inf
                            ori = no_text_images[i]*255
                            # print('rec.shape',rec.shape)
                            # print('rec.max()',rec.max())
                            # print('ori.max()',ori.max())
                            # mse = F.mse_loss(rec, ori)
                            mse = torch.mean((rec - ori) ** 2)
                            # print('mse',mse)
                            max_value = 255.0  # 假设像素值范围在0到1之间
                            psnr = 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(mse)
                            # print('psnr',psnr)
                            # assert psnr > 0
                            psnr_list_sum += psnr
                            psnr_num_sum += 1
                if misc.is_main_process():
                    if psnr_num_sum != 800:
                        print(psnr_num_sum)
                        assert psnr_num_sum == 800
                    psnr = psnr_list_sum/psnr_num_sum
                    if psnr > best_psnr:
                        torch.save(checkpoint, os.path.join(checkpoint_dir, f"best.pth"))
                        best_psnr = psnr
                    print('current_psnr:',psnr)
                    print('best_psnr:',best_psnr)
                            
            model.train()
        lr_scheduler.step()

    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    if misc.is_main_process():
        model_name = "/final_epoch_" + str(epoch_num) + ".pth"
        torch.save(model.state_dict(), args.output + model_name)


def inference_on_dataset(model, data_loader, data_name, evaluator, args):
    print("Start inference on {}, {} batches".format(data_name, len(data_loader)))
    num_devices = misc.get_world_size()
    total = len(data_loader)
    evaluator.reset()
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    start_data_time = time.perf_counter()
    for idx_val, data_val in enumerate(data_loader):
        inputs_val, labels_ori = data_val['image'], data_val['ori_label']
        ignore_mask = data_val.get('ignore_mask', None)
        if torch.cuda.is_available():
            labels_ori = labels_ori.cuda()
        batched_input = []
        for b_i in range(len(inputs_val)):
            dict_input = dict()
            dict_input['image'] = inputs_val[b_i].to(model.device).contiguous()
            dict_input['original_size'] = labels_ori[b_i].shape[-2:]
            batched_input.append(dict_input)

        total_data_time += time.perf_counter() - start_data_time
        if idx_val == num_warmup:
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0

        start_compute_time = time.perf_counter()
        with torch.no_grad():
            up_masks_logits, up_masks, iou_output, hr_masks_logits, hr_masks, hr_iou_output = model(
                batched_input, multimask_output=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_compute_time += time.perf_counter() - start_compute_time

        start_eval_time = time.perf_counter()
        evaluator.process(up_masks, hr_masks, labels_ori, ignore_mask)
        total_eval_time += time.perf_counter() - start_eval_time

        iters_after_start = idx_val + 1 - num_warmup * int(idx_val >= num_warmup)
        data_seconds_per_iter = total_data_time / iters_after_start
        compute_seconds_per_iter = total_compute_time / iters_after_start
        eval_seconds_per_iter = total_eval_time / iters_after_start
        total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

        if (idx_val+1) % 20 == 0:
            eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx_val - 1)))
            print(
                f"Inference done [{idx_val + 1}]/[{total}]. ",
                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. ",
                f"Inference: {compute_seconds_per_iter:.4f} s/iter. ",
                f"Eval: {eval_seconds_per_iter:.4f} s/iter. ",
                f"Total: {total_seconds_per_iter:.4f} s/iter. ",
                f"ETA={eta}"
            )
        start_data_time = time.perf_counter()

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    print(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    if results is None:
        results = {}

    return results


def evaluate(args, model, valid_dataloaders, valid_datasets_names):
    model.eval()
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        valid_dataset_name = valid_datasets_names[k]
        evaluator = Evaluator(valid_dataset_name, args, True)
        print('============================')
        results_k = inference_on_dataset(model, valid_dataloader, valid_dataset_name, evaluator, args)
        print("Evaluation results for {}:".format(valid_dataset_name))
        for task, res in results_k.items():
            if '_hr' not in task:
                print(f"copypaste: {task}={res}, {task}_hr={results_k[task+'_hr']}")
        print('============================')
        test_stats.update({valid_dataset_name: results_k})

    return test_stats


if __name__ == "__main__":

    # train
    totaltext_train = {
        "name": "TotalText-train",
        "im_dir": "./datasets/TotalText/Images/Train",
        "gt_dir": "./datasets/TotalText/groundtruth_pixel/Train",
        "im_ext": ".jpg",
        "gt_ext": ".jpg",
    }
    hiertext_train = {
        "name": "HierText-train",
        "im_dir": "./datasets/HierText/train",
        "gt_dir": "./datasets/HierText/train_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png",
        "json_dir": "./datasets/HierText/train_shrink_vert.json"
    }
    textseg_train = {
        "name": "TextSeg-train",
        "im_dir": "./datasets/TextSeg/train_images",
        "gt_dir": "./datasets/TextSeg/train_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/COCO_TS_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_hier = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/hier-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_tt = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/tt-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    cocots_train_textseg = {
        "name": "COCO_TS-train",
        "im_dir": "./datasets/COCO_TS/train_images",
        "gt_dir": "./datasets/COCO_TS/textseg-model_labels",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    flickrst_train = {
        "name": "FlickrST-train",
        "im_dir": "./datasets/FlickrST/train/image",
        "no_text_im_dir": "./datasets/FlickrST/train/label",
        "gt_dir": "./datasets/FlickrST/train/mask",
        "im_ext": ".jpg",
        "no_text_im_ext": ".jpg",
        "gt_ext": ".png",
        "word_dir": "./datasets/FlickrST/train/word_box",
        "word_ext": ".txt"
    }

    train_dataset_map = {
        'totaltext_train': totaltext_train,
        'hiertext_train': hiertext_train,
        'textseg_train': textseg_train,
        'cocots_train': cocots_train,
        'cocots_train_hier': cocots_train_hier,
        'cocots_train_tt': cocots_train_tt,
        'cocots_train_textseg': cocots_train_textseg,
        'flickrst_train': flickrst_train
    }

    # validation and test
    totaltext_test = {
        "name": "TotalText-test",
        "im_dir": "./datasets/TotalText/Images/Test",
        "gt_dir": "./datasets/TotalText/groundtruth_pixel/Test",
        "im_ext": ".jpg",
        "gt_ext": ".jpg"
    }
    hiertext_val = {
        "name": "HierText-val",
        "im_dir": "./datasets/HierText/validation",
        "gt_dir": "./datasets/HierText/validation_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    hiertext_test = {
        "name": "HierText-test",
        "im_dir": "./datasets/HierText/test",
        "gt_dir": "./datasets/HierText/test_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    textseg_val = {
        "name": "TextSeg-val",
        "im_dir": "./datasets/TextSeg/val_images",
        "gt_dir": "./datasets/TextSeg/val_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    textseg_test = {
        "name": "TextSeg-test",
        "im_dir": "./datasets/TextSeg/test_images",
        "gt_dir": "./datasets/TextSeg/test_gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }

    flickrst_test = {
        "name": "FlickrST-test",
        "im_dir": "./datasets/FlickrST/test/image",
        "no_text_im_dir": "./datasets/FlickrST/test/label",
        "gt_dir": "./datasets/FlickrST/test/mask",
        "im_ext": ".jpg",
        "no_text_im_ext": ".jpg",
        "gt_ext": ".png",
        "word_dir": "./datasets/FlickrST/test/word_box",
        "word_ext": ".txt"
    }

    val_dataset_map = {
        'totaltext_test': totaltext_test,
        'hiertext_val': hiertext_val,
        'hiertext_test': hiertext_test,
        'textseg_val': textseg_val,
        'textseg_test': textseg_test,
        'flickrst_test': flickrst_test
    }

    train_datasets = []
    val_datasets = []
    args = get_args_parser()

    for ds_name in args.train_datasets:
        train_datasets.append(train_dataset_map[ds_name])
    for ds_name in args.val_datasets:
        val_datasets.append(val_dataset_map[ds_name])

    main(train_datasets, val_datasets, args)
