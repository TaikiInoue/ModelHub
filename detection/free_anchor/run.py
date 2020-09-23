from importlib import import_module

import nablasian.datasets
import torch
from nablasian.datasets import Composite
from nablasian.solvers.detection.decoders import AnchorBoxDecoder
from nablasian.solvers.detection.matchers import IoUBasedMatcher
from nablasian.solvers.detection.nms import NMS
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from PIL import Image
from torch import Tensor
from torch.nn import Module


def load_model(cfg: DictConfig) -> Module:

    module = import_module(cfg.model.module)
    cls = getattr(module, cfg.model.name)
    model = cls(**cfg.model.args)
    model = model.to(cfg.device)
    model.load_state_dict(torch.load(cfg.model.pretrained, map_location=cfg.device))
    model.eval()
    return model


def load_anchor_box_generators(cfg: DictConfig) -> list:

    module = import_module(cfg.anchor_box_generators.module)
    cls = getattr(module, cfg.anchor_box_generators.name)

    anchor_box_generators = []
    for (size, stride) in zip(cfg.anchor_box_generators.size, cfg.anchor_box_generators.stride):
        anchor_box_generators.append(cls(size, stride, **cfg.anchor_box_generators.args))

    return anchor_box_generators


def load_transform(cfg: DictConfig) -> Composite:

    transform = []
    for transform_cfg in cfg.transform:
        cls = getattr(nablasian.datasets, transform_cfg.name)
        if transform_cfg.args:
            transform.append(cls(**transform_cfg.args))
        else:
            transform.append(cls())

    return Composite(transform)


def pre_processing(cfg: DictConfig, transform: Composite, img_path: str) -> Tensor:

    data = {}
    data["Image"] = Image.open(img_path).convert("RGB")
    data = transform(data)
    img = data["Image"]
    img = img.unsqueeze(0)
    img = img.to(cfg.device)
    return img


def inference(cfg: DictConfig, model: Module, img: Tensor, anchor_box_generators: list):

    meta_info = {"img_shape": [img.size()[-2:]]}

    with torch.no_grad():
        (mb_reg_logits, mb_cls_logits, meta_info) = model(img, meta_info)

    mb_reg_logits = torch.cat(mb_reg_logits, dim=1)
    mb_cls_logits = torch.cat(mb_cls_logits, dim=1)

    mb_cls_logits = torch.sigmoid(mb_cls_logits)

    anchor_boxes = []
    for (i, stride) in enumerate(cfg.anchor_box_generators.stride):
        anchor_boxes.append(anchor_box_generators[i](meta_info["feat_shape"][i]))
    anchor_boxes = torch.cat(anchor_boxes, dim=0)

    # Remove invalid entries
    matcher = IoUBasedMatcher(cfg.post_processing.pre_mean, cfg.post_processing.pre_std)
    matcher.set_items(anchor_boxes)
    mask = matcher.area > 0
    mb_reg_logits = mb_reg_logits[:, mask]
    mb_cls_logits = mb_cls_logits[:, mask]
    anchor_boxes = anchor_boxes[mask].unsqueeze(0)

    # Decode
    decoder = AnchorBoxDecoder(cfg.post_processing.pre_mean, cfg.post_processing.pre_std)
    (cl_pre_bboxes, cl_cls_logits) = decoder(
        [mb_reg_logits], [mb_cls_logits], [anchor_boxes], meta_info["img_shape"]
    )

    # NMS
    nms = NMS(
        cfg.post_processing.score_th,
        cfg.post_processing.iou_th,
        cfg.post_processing.use_sigmoid,
        None,
        cfg.post_processing.pre_nms,
        cfg.post_processing.post_nms,
    )

    (cl_pre_bboxes, cl_cls_logits) = nms(cl_pre_bboxes, cl_cls_logits)

    mb_pre_bboxes = cl_pre_bboxes[0][0]
    mb_cls_logits = cl_cls_logits[0][0]
    return mb_pre_bboxes, mb_cls_logits


if __name__ == "__main__":

    cfg = OmegaConf.load("./config.yaml")
    model = load_model(cfg)
    anchor_box_generators = load_anchor_box_generators(cfg)
    transform = load_transform(cfg)
    img = pre_processing(cfg, transform, "./61_8_1.jpg")
    mb_pre_bboxes, mb_cls_logits = inference(cfg, model, img, anchor_box_generators)
    print(mb_pre_bboxes, mb_cls_logits)
