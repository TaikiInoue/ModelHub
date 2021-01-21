from importlib import import_module
from typing import Any, Callable

import torch
from nablasian.datasets import Composite
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torch.nn import Module


class FreeAnchor:
    def __init__(self, yaml_path: str) -> None:

        self.cfg = OmegaConf.load(yaml_path)
        self.model = self._init_model()
        self.anchor_generator = self._init_anchor_generator()
        self.transform = self._init_transforms()
        self.nms = self._init_nms()
        self.coder = self._init_coder

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)

    def _init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        model = attr(**cfg.get("args", {}))
        # TODO(inoue): Should move .pth path to yaml
        model.load_state_dict(torch.load("free_anchor/RX101-FA-HV1-Head.pth", map_location="cuda"))
        model = model.to("cuda")
        return model.eval()

    def _init_anchor_generator(self) -> Module:

        cfg = self.cfg.anchor_generator
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_transforms(self) -> Composite:

        transforms = []
        for cfg in self.cfg.transforms:
            attr = self._get_attr(cfg.name)
            transforms.append(attr(**cfg.get("args", {})))
        return Composite(transforms)

    def _init_nms(self) -> Callable:

        cfg = self.cfg.nms
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_coder(self) -> Callable:

        cfg = self.cfg.coder
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def pre_processing(self, img_path: str) -> Tensor:

        data = {}
        data["Image"] = Image.open(img_path).convert("RGB")
        data = self.transform(data)
        img = data["Image"]
        img = img.unsqueeze(0)
        img = img.to(self.cfg.device)
        return img

    def inference(self, mb_imgs: Tensor) -> None:

        with torch.no_grad():
            (cl_reg_preds, cl_cls_preds, cl_feats) = self.model(mb_imgs)

        meta_info = [{"img_shape": [mb_imgs.size()[-2:]]}]
        cl_anchor_boxes, cl_masks = self.anchor_generator(mb_imgs, cl_feats, meta_info)

        mb_bboxes = []
        mb_scores = []
        mb_labels = []
        for i in range(len(meta_info)):

            mb_reg_preds = []
            mb_cls_preds = []
            mb_anchor_boxes = []
            for (r, c, a, m) in zip(cl_reg_preds, cl_cls_preds, cl_anchor_boxes, cl_masks):
                mb_reg_preds.append(r[i][m[i]])
                mb_cls_preds.append(c[i][m[i]])
                mb_anchor_boxes.append(a[i][m[i]])

            result = self.nms(
                mb_reg_preds,
                mb_cls_preds,
                mb_anchor_boxes,
                meta_info[0]["img_shape"],
                self.coder,
                False,
            )

            mb_bboxes.append(result[0])
            mb_scores.append(result[1])
            mb_labels.append(result[2])

        return (mb_bboxes, mb_scores, mb_labels)
