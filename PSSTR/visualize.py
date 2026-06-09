import random
import numpy as np
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchinfo import summary
from matplotlib import pyplot as plt


def process_prompt(predictor, point_coords, point_labels, box, mask):
    coords_torch, labels_torch, box_torch = None, None, None
    if point_coords is not None:
        assert (
            point_labels is not None
        ), "point_labels must be supplied if point_coords is supplied."
        point_coords = predictor.transform.apply_coords(point_coords, predictor.original_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=predictor.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
        coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
        # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        print("points >>> ", coords_torch.shape, labels_torch.shape)
    if box is not None:
        box = predictor.transform.apply_boxes(box, predictor.original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=predictor.device)
        if box_torch.shape[0] == 1:
            box_torch = box_torch[None, :]
        print("\ninput_box in torch: ", box_torch.shape)
        # box_torch = box_torch[None, :]
    if mask is not None:
        h_idx, w_idx = mask.nonzero()
        total_points = h_idx.shape[0]
        pos = random.sample(range(0, total_points), min(total_points, num_point_per_area))
        h_idx = h_idx[pos]
        w_idx = w_idx[pos]
        point_coords = np.column_stack((w_idx, h_idx)) # 将mask里的points采样压缩
        point_labels = np.ones(point_coords.shape[0])
        point_coords = predictor.transform.apply_coords(point_coords, predictor.original_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=predictor.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    
    return coords_torch, labels_torch, box_torch


def visualization(img_name, predictor, sam_model, point_coords, point_labels, box, mask):
    nodes, _ = get_graph_node_names(sam_model.unimask_decoder)
    for node in nodes:
        print(node)

    features = [
        # 'output_hypernetworks_mlps.0.layers.2',
        # 'output_hypernetworks_mlps.1.layers.2',
        # 'output_upscaling.4'
        # 'view',
        # 'view_2',
        'getitem_19',
    ]

    # return_nodes参数就是返回对应的输出
    feature_extractor = create_feature_extractor(sam_model.unimask_decoder, return_nodes=features)

    point_coords, point_labels, boxes = process_prompt(predictor, point_coords, point_labels, box, mask)
    if point_coords is not None:
        points = (point_coords, point_labels)
    else:
        points = None
    point_embeddings, _ = predictor.model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=None
    )

    out = feature_extractor(
        image_embeddings=predictor.features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=point_embeddings,
        dense_prompt_embeddings=None,
        multimask_output=True
    )

    tot_row = 4
    print(out.keys())
    for feature in features:
        # 绘制多个子图
        print(f'[{feature}] layer output shape >>> ', out[feature].shape)

        B, C, H, W = out[feature].shape
        for b in range(B):
            rows = tot_row if C >= tot_row else 1
            cols = C // rows
            fig, axes = plt.subplots(rows, cols, figsize=(80, 50))

            for i in range(C):
                row = i // cols
                col = i % cols

                if row == 0 and col == 0:
                    ax = axes
                elif rows > 1 and cols > 1:
                    ax = axes[row, col]
                elif rows <= 1:
                    ax = axes[col]
                elif cols <= 1:
                    ax = axes[row]
                ax.imshow(out[feature][0][i].cpu().detach().numpy(), cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Feature Map {i}')
            
            # 缩小间距
            plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
            plt.subplots_adjust(top=0.90)
            plt.savefig("visual_result/" + img_name + "_" + feature + ".png", bbox_inches='tight')
        # plt.axis('off')
        # out = out[feature].sum(1).sigmoid().squeeze(0).detach().cpu().numpy()
        # plt.imshow(out)
    