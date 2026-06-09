# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import random

from hi_sam import HiSam
from dinet import EraseDecoder
from typing import Optional, Tuple, List
from hi_sam.data.transforms import ResizeLongestSide


class ErasePredictor:
    def __init__(
        self,
        sam_model: HiSam,
        erase_model: EraseDecoder
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.sam_model = sam_model
        self.erase_model = erase_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        # import pdb;pdb.set_trace()
        if image_format != self.sam_model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        # import pdb;pdb.set_trace()
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.sam_model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.sam_model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.sam_model.preprocess(transformed_image)
        self.features = self.sam_model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        multimask_output: bool = False,
        return_logits: bool = False,
        promptable: bool = False,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        words: Optional[List[str]] = None,
        num_point_per_area: int = 10
    ):
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if promptable:
            if point_coords is not None:
                assert (
                    point_labels is not None
                ), "point_labels must be supplied if point_coords is supplied."
                point_coords = self.transform.apply_coords(point_coords, self.original_size)
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                # coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                # print("\npoints >>> ", coords_torch.shape, labels_torch.shape)
            if box is not None:
                box = self.transform.apply_boxes(box, self.original_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
                if box_torch.shape[0] == 1:
                    box_torch = box_torch[None, :]
                # print("\ninput_box in torch >>>> ", box_torch.shape)
            if mask is not None:
                h_idx, w_idx = mask.nonzero()
                total_points = h_idx.shape[0]
                # print('total_points',total_points,num_point_per_area)
                pos = random.sample(range(0, total_points), min(total_points, num_point_per_area))
                h_idx = h_idx[pos]
                w_idx = w_idx[pos]
                point_coords = np.column_stack((w_idx, h_idx)) # 从mask里采样points，并压缩
                # print(point_coords.shape)
                point_labels = np.ones(point_coords.shape[0])
                point_coords = self.transform.apply_coords(point_coords, self.original_size)
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            
            pr_masks, pr_iou, pr_word_masks, image_hr, mask_hr = self.predict_torch(
                multimask_output,
                return_logits=return_logits,
                promptable=True,
                point_coords=coords_torch,
                point_labels=labels_torch,
                boxes=box_torch,
                words=words
            )
            pr_masks_np = pr_masks.detach().cpu().numpy()
            pr_iou_np = pr_iou.detach().cpu().numpy()
            pr_word_masks_np = pr_word_masks.detach().cpu().numpy()
            image_hr_np = (image_hr.detach().cpu().numpy() * 255).astype(np.uint8)
            mask_hr_np = mask_hr.detach().cpu().numpy()
            return (pr_masks_np, pr_iou_np, pr_word_masks_np, image_hr_np, mask_hr_np)
        else:
            raise RuntimeError("--promptable must be set.")

    @torch.no_grad()
    def predict_torch(
            self,
            multimask_output: bool = False,
            return_logits: bool = False,
            promptable: bool = False,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            boxes: Optional[torch.Tensor] = None,
            words: Optional[List[str]] = None,
    ):
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
            
        if promptable:
            if point_coords is not None:
                points = (point_coords, point_labels)
            else:
                points = None
            point_embeddings, _ = self.sam_model.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=None
            )
            sparse_prompt_embeddings = point_embeddings
            # print(sparse_prompt_embeddings.shape) #[1, 2, 256]
            if words is not None:
                # print('\nword >>>> ', words)
                words_embedding = self.sam_model.word_embedding(words)
                # lg20230325 
                words_embedding = self.sam_model.word_chan_proj(words_embedding)
                sparse_prompt_embeddings = words_embedding
            # print(sparse_prompt_embeddings.shape) #[2, 24, 256]
            

            pr_masks, pr_iou_pred, pr_word_masks, mask_embeddings = self.sam_model.unimask_decoder(
                image_embeddings=self.features,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                multimask_output=True
            )

            # print(self.features.shape, mask_embeddings.shape) # torch.Size([1, 256, 64, 64]) torch.Size([1, 256, 64, 64])

            # for image_emb, mask_emb in zip(self.features, mask_embeddings):
            #     B = mask_emb.shape[0]
            #     image_emb = torch.repeat_interleave(image_emb.unsqueeze(0), B, dim=0)
            #     print(image_emb.shape)
            #     erased_images, output_masks = self.erase_model(image_embeddings=image_emb, mask_embeddings=mask_emb)

            B = mask_embeddings.shape[0]
            image_emb = torch.repeat_interleave(self.features, B, dim=0)
            # print(image_emb.shape) # torch.Size([1, 256, 64, 64])
            erased_images, output_masks = self.erase_model(image_embeddings=image_emb, mask_embeddings=mask_embeddings)

            # erased_images, output_masks = self.erase_model(
            #     image_embeddings=self.features,
            #     mask_embeddings=mask_embeddings
            # )

            erased_image_hr = erased_images[-1]
            output_mask_hr = output_masks[-1]
            # print(type(erased_image_hr)) # <class 'torch.Tensor'>

            # Upscale the masks to the original image resolution
            pr_masks = self.sam_model.postprocess_masks(pr_masks[:, :, :, :], self.input_size, self.original_size)
            pr_word_masks = self.sam_model.postprocess_masks(pr_word_masks, self.input_size, self.original_size)
            erased_image_hr = self.sam_model.postprocess_masks(erased_image_hr[:, :, :, :], self.input_size, self.original_size)
            output_mask_hr = self.sam_model.postprocess_masks(output_mask_hr, self.input_size, self.original_size)

            # print(pr_masks.shape, erased_image_hr.shape, output_mask_hr.shape) # torch.Size([1, 1, 427, 640]) torch.Size([1, 3, 427, 640]) torch.Size([1, 1, 427, 640])
            # pr_masks, erased_image_hr, output_mask_hr = pr_masks[-1], erased_image_hr[-1], output_mask_hr[-1]
            # pr_masks, erased_image_hr, output_mask_hr = pr_masks[0], erased_image_hr[0], output_mask_hr[0]
            pr_masks, final_erased_image_hr, final_output_mask_hr = pr_masks[0], erased_image_hr[0], output_mask_hr[0]
            final_output_mask_hr = (final_output_mask_hr.sigmoid() + 0.5).int()
            final_erased_image_hr = torch.clamp(final_erased_image_hr, 0, 1)
            n_prompt=erased_image_hr.shape[0]
            for prompt_idx in range(1,n_prompt):
                new_output_mask_hr = (output_mask_hr[prompt_idx].sigmoid() + 0.5).int()
                new_erased_image_hr = torch.clamp(erased_image_hr[prompt_idx], 0, 1)
                final_output_mask_hr = (1-new_output_mask_hr)*final_output_mask_hr + new_output_mask_hr*new_output_mask_hr
                final_erased_image_hr = (1-new_output_mask_hr)*final_erased_image_hr + new_output_mask_hr*new_erased_image_hr
            # output_mask_hr = output_mask_hr.sigmoid().float()
            # output_mask_hr = (output_mask_hr.sigmoid() + 0.5).int()
            # erased_image_hr = torch.clamp(erased_image_hr, 0, 1)


            # if not return_logits:
            #     # pr_masks = pr_masks > self.sam_model.mask_threshold
            #     # pr_word_masks = pr_word_masks > self.sam_model.mask_threshold
            #     # output_mask_hr = output_mask_hr > self.sam_model.mask_threshold

            #     # erased_image_hr = erased_image_hr > self.sam_model.mask_threshold

            #     from torchvision.transforms import ToPILImage
            #     # 将 tensor 转为 PIL 图像
            #     to_pil = ToPILImage()
            #     pil_image = to_pil(erased_image_hr.squeeze(0))
            #     pil_image.save('output_image.png')

            #     pil_mask = to_pil(output_mask_hr.sigmoid().float().squeeze(0))
            #     pil_mask.save('output_mask.png')

            #     pil_pr_masks = to_pil(pr_masks.sigmoid().float().squeeze(0))
            #     pil_pr_masks.save('output_pr_masks.png')

            #     # exit(0)

            # return pr_masks, pr_iou_pred, pr_word_masks, erased_image_hr, output_mask_hr
            pr_word_masks = pr_word_masks > self.sam_model.mask_threshold
            return pr_masks, pr_iou_pred, pr_word_masks, final_erased_image_hr, final_output_mask_hr
        else:
            raise RuntimeError("--promptable must be set.")

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.sam_model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
