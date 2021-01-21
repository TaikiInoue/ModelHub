import math


# === Backbone ================================================================

c.ResNet.num_base_ch = 64

c.ResNet.num_base_ch = 64

c.ResNet.stem_cfg = {
    "requires_grad": True,
    "norm": {"name": "SynchronizedBatchNorm2d", "requires_grad": True},
    "nonlinear": {"name": "ReLU"},
    "initializer": {"name": "StandardInitializer"},
}

c.ResNet.stage1_cfg = {
    "requires_grad": True,
    "norm": {"name": "SynchronizedBatchNorm2d", "requires_grad": True},
    "nonlinear": {"name": "ReLU"},
    "initializer": {"name": "StandardInitializer"},
}

c.ResNet.stage2_cfg = {
    "requires_grad": True,
    "norm": {"name": "SynchronizedBatchNorm2d", "requires_grad": True},
    "nonlinear": {"name": "ReLU"},
    "initializer": {"name": "StandardInitializer"},
}

c.ResNet.stage3_cfg = {
    "requires_grad": True,
    "norm": {"name": "SynchronizedBatchNorm2d", "requires_grad": True},
    "nonlinear": {"name": "ReLU"},
    "initializer": {"name": "StandardInitializer"},
}

c.ResNet.stage4_cfg = {
    "requires_grad": True,
    "norm": {"name": "SynchronizedBatchNorm2d", "requires_grad": True},
    "nonlinear": {"name": "ReLU"},
    "initializer": {"name": "StandardInitializer"},
}


# === FPN =====================================================================


c.FPN.backbone_cfg = {
    "name": "ResNeXt101",
    "num_classes": 1000,
    "use_feature_only": [0, 1, 2, 3],
    "pretrain": "./free_anchor/ResNeXt101_32x8d.pth",
    "use_extra_conv": False
}

c.FPN.inner_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "Identity",
    },
    "initializer": {
        "name": "XavierInitializer"
    }
}

c.FPN.layer_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "Identity",
    },
    "initializer": {
        "name": "XavierInitializer"
    }
}

"""
c.FPN.p6_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "Identity",
    },
    "initializer": {
        "name": "XavierInitializer"
    }
}

c.FPN.p7_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "ReLU",
        "inplace": True
    },
    "initializer": {
        "name": "XavierInitializer"
    },
    "order": "acn"
}
"""


# === Main Module =============================================================


c.RetinaNet.backbone_cfg = {
    "name": "FPN",
    "num_out_ch": 256,
    "use_feature_only": [0, 1, 2, 3]
}

c.RetinaNet.reg_sub_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "ReLU",
        "inplace": True
    },
    "initializer": {
        "name": "GaussianInitializer",
        "mean": 0.0,
        "std": 0.01
    }
}

c.RetinaNet.reg_head_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "Identity"
    },
    "initializer": {
        "name": "GaussianInitializer",
        "mean": 0.0,
        "std": 0.01
    }
}

c.RetinaNet.cls_sub_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "ReLU",
        "inplace": True
    },
    "initializer": {
        "name": "GaussianInitializer",
        "mean": 0.0,
        "std": 0.01
    }
}

c.RetinaNet.cls_head_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "Identity"
    },
    "initializer": {
        "name": "GaussianInitializer",
        "c": -math.log((1 - 0.01) / 0.01)
    }
}
