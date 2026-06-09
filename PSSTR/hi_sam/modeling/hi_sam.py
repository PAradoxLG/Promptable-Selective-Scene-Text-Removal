# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, HiDecoder
from .prompt_encoder import PromptEncoder
from .text_align import WordEmbedding

class HiSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        # for n, p in self.image_encoder.named_parameters():
        #     if "Adapter" not in n:
        #         p.requires_grad = False
        # print("Freeze image encoder.")

        self.prompt_encoder = prompt_encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

        self.word_prompt = False
        self.word_embedding = None
        self.word_chan_proj = None

        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.promptable = False
        self.unimask_decoder = None

        self.erase_mode = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
    ):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        image_embeddings = self.image_encoder(input_images)
        if self.promptable:
            pr_masks_logits = []
            pr_iou_preds = []
            pr_word_masks_logits = []
            mask_embeddings = []


        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # low_res_masks, high_res_masks, iou_pred, iou_pred_hr = self.mask_decoder(
            #     image_embeddings=curr_embedding.unsqueeze(0),
            #     image_pe=self.prompt_encoder.get_dense_pe(),
            #     sparse_prompt_embeddings=sparse_embeddings.unsqueeze(0),
            #     multimask_output=multimask_output
            # )
            # iou_preds.append(iou_pred)
            # iou_preds_hr.append(iou_pred_hr)
            # upscaled_masks = self.postprocess_masks(
            #     low_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            # high_res_masks = self.postprocess_masks(
            #     high_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            # up_masks_logits.append(upscaled_masks)
            # up_masks.append(upscaled_masks > self.mask_threshold)
            # hr_masks_logits.append(high_res_masks)
            # hr_masks.append(high_res_masks > self.mask_threshold)

            if self.promptable:
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])
                else:
                    points = None

                sparse_embeddings, _ = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )

                if "word" in image_record:
                    words_one_image = image_record['word']
                    sparse_embeddings = self.word_embedding(words_one_image)
                    # lg20250305
                    # print('word#$'*10,sparse_embeddings.shape,torch.abs(sparse_embeddings).mean())
                    sparse_embeddings = self.word_chan_proj(sparse_embeddings)
                    # print('proj#$'*10,sparse_embeddings.shape,torch.abs(sparse_embeddings).mean())
                    # print('proj#$'*10,sparse_embeddings.shape,sparse_embeddings.mean())
                #     # lg20250307
                # else:
                #     print('else#$'*10,sparse_embeddings.shape,torch.abs(sparse_embeddings).mean())

                pr_masks, pr_iou_pred, pr_word_masks, mask_embedding = self.unimask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    # dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True
                )

                pr_masks_logits.append(pr_masks)
                pr_iou_preds.append(pr_iou_pred)
                pr_word_masks_logits.append(pr_word_masks)
                mask_embeddings.append(mask_embedding)


        if self.promptable:
            pr_masks_logits = torch.cat(pr_masks_logits, dim=0)
            pr_iou_preds = torch.cat(pr_iou_preds, dim=0)
            pr_word_masks_logits = torch.cat(pr_word_masks_logits, dim=0)

            if self.erase_mode: # 擦除模式，返回image_embeddings和mask_embeddings
                return pr_masks_logits, pr_iou_preds, pr_word_masks_logits, image_embeddings, mask_embeddings

            return pr_masks_logits, pr_iou_preds, pr_word_masks_logits

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x