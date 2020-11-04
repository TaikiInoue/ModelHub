from importlib import import_module
from time import time

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


class FreeAnchor:

    def __init__(self, yaml_path: str):

        self.cfg = OmegaConf.load(yaml_path)
        self.model = self.init_model()
        self.anchor_box_generators = self.init_anchor_box_generators()
        self.transform = self.init_transform()


    def init_model(self) -> Module:

        module = import_module(self.cfg.model.module)
        cls = getattr(module, self.cfg.model.name)
        model = cls(**self.cfg.model.args)
        model = model.to(self.cfg.device)
        model.load_state_dict(torch.load(self.cfg.model.pretrained, map_location=self.cfg.device))
        model.eval()
        return model


    def init_anchor_box_generators(self) -> list:

        module = import_module(self.cfg.anchor_box_generators.module)
        cls = getattr(module, self.cfg.anchor_box_generators.name)

        anchor_box_generators = []
        for (size, stride) in zip(self.cfg.anchor_box_generators.size, self.cfg.anchor_box_generators.stride):
            anchor_box_generators.append(cls(size, stride, **self.cfg.anchor_box_generators.args))

        return anchor_box_generators


    def init_transform(self) -> Composite:

        transform = []
        for transform_cfg in self.cfg.transform:
            cls = getattr(nablasian.datasets, transform_cfg.name)
            if transform_cfg.args:
                transform.append(cls(**transform_cfg.args))
            else:
                transform.append(cls())

        return Composite(transform)


    def pre_processing(self, img_path: str) -> Tensor:

        data = {}
        data["Image"] = Image.open(img_path).convert("RGB")
        data = self.transform(data)
        img = data["Image"]
        img = img.unsqueeze(0)
        img = img.to(self.cfg.device)
        return img


    def inference(self, img: Tensor):
        
        t0 = time()
        meta_info = {"img_shape": [img.size()[-2:]]}

        with torch.no_grad():
            (mb_reg_logits, mb_cls_logits, meta_info) = self.model(img, meta_info)

        t1 = time()
        mb_reg_logits = torch.cat(mb_reg_logits, dim=1)
        mb_cls_logits = torch.cat(mb_cls_logits, dim=1)

        mb_cls_logits = torch.sigmoid(mb_cls_logits)

        anchor_boxes = []
        for (i, stride) in enumerate(self.cfg.anchor_box_generators.stride):
            anchor_boxes.append(self.anchor_box_generators[i](meta_info["feat_shape"][i]))
        anchor_boxes = torch.cat(anchor_boxes, dim=0)
        
        t2 = time()
        # Remove invalid entries
        matcher = IoUBasedMatcher(self.cfg.post_processing.pre_mean, self.cfg.post_processing.pre_std)
        matcher.set_items(anchor_boxes)
        mask = matcher.area > 0
        mb_reg_logits = mb_reg_logits[:, mask]
        mb_cls_logits = mb_cls_logits[:, mask]
        anchor_boxes = anchor_boxes[mask].unsqueeze(0)
        
        t3 = time()
        # Decode
        decoder = AnchorBoxDecoder(self.cfg.post_processing.pre_mean, self.cfg.post_processing.pre_std)
        (cl_pre_bboxes, cl_cls_logits) = decoder(
            [mb_reg_logits], [mb_cls_logits], [anchor_boxes], meta_info["img_shape"]
        )
        
        t4 = time()
        # NMS
        nms = NMS(
            self.cfg.post_processing.score_th,
            self.cfg.post_processing.iou_th,
            self.cfg.post_processing.use_sigmoid,
            None,
            self.cfg.post_processing.pre_nms,
            self.cfg.post_processing.post_nms,
        )

        (cl_pre_bboxes, cl_cls_logits) = nms(cl_pre_bboxes, cl_cls_logits)

        mb_pre_bboxes = cl_pre_bboxes[0][0]
        mb_cls_logits = cl_cls_logits[0][0]

        t5 = time()

        print("inference: ", t1 - t0)
        print("generate anchor boxes: ", t2 - t1)
        print("remove invalid entries", t3 - t2)
        print("decode", t4 - t3)
        print("nms", t5 - t4)

        return mb_pre_bboxes, mb_cls_logits
