import numpy as np

import torch
from torch import nn

from hi_sam.modeling.hi_sam import HiSam

from typing import Any, Dict, List, Tuple

class UniStR(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        mask_extractor: HiSam,
        erase_decoder: nn.Module
    ) -> None:
        super().__init__()
        self.mask_extractor = mask_extractor

        self.erase_decoder = erase_decoder

    @property
    def device(self) -> Any:
        return self.mask_extractor.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ):
        
        if not self.mask_extractor.erase_mode:
            return self.mask_extractor(batched_input)

        pr_masks_logits, pr_iou_preds, pr_word_masks_logits, image_embeddings, mask_embeddings = self.mask_extractor(batched_input)

        # for i in range(len(mask_embeddings)):
        #     print('mask_embeddings[i].shape', mask_embeddings[i].shape)

        outputs_list, masks_list = [], []
        
        for image_record, image_emb, mask_emb in zip(batched_input, image_embeddings, mask_embeddings):
            B = mask_emb.shape[0]
            image_emb = torch.repeat_interleave(image_emb.unsqueeze(0), B, dim=0)

            outputs, masks = self.erase_decoder(image_emb, mask_emb)

            # final_output = self.mask_extractor.postprocess_masks(outputs[-1], image_record['input_size'], image_record['original_size'])
            # final_mask = self.mask_extractor.postprocess_masks(masks[-1], image_record['input_size'], image_record['original_size'])

            # outputs.append(final_output)
            # masks.append(final_mask)
            outputs_list.append(outputs)
            masks_list.append(masks)


        return pr_masks_logits, pr_iou_preds, pr_word_masks_logits, outputs_list, masks_list