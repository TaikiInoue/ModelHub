# === Backbone ================================================================

c.ResNet.num_base_ch = 64

c.ResNet.stem_cfg = {
    "requires_grad": False,
    "norm": {
        "name": "BatchNorm2d"
    },
    "nonlinear": {
        "name": "ReLU",
    },
    "initializer": {
        "name": "StandardInitializer"
    }
}

c.ResNet.stage1_cfg = {
    "requires_grad": False,
    "norm": {
        "name": "BatchNorm2d"
    },
    "nonlinear": {
        "name": "ReLU",
    },
    "initializer": {
        "name": "StandardInitializer"
    }
}

c.ResNet.stage2_cfg = {
    "requires_grad": True,
    "norm": {
        "name": "BatchNorm2d"
    },
    "nonlinear": {
        "name": "ReLU",
    },
    "initializer": {
        "name": "StandardInitializer"
    }
}

c.ResNet.stage3_cfg = {
    "requires_grad": True,
    "norm": {
        "name": "BatchNorm2d"
    },
    "nonlinear": {
        "name": "ReLU",
    },
    "initializer": {
        "name": "StandardInitializer"
    }
}

c.ResNet.stage4_cfg = {
    "requires_grad": True,
    "norm": {
        "name": "BatchNorm2d"
    },
    "nonlinear": {
        "name": "ReLU",
    },
    "initializer": {
        "name": "StandardInitializer"
    }
}

c.ResNet.fc_cfg = {
    "requires_grad": False,
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "Identity",
    },
    "initializer": {
        "name": "LinearInitializer"
    }
}


# === Main Module =============================================================


c.RetinaNet.backbone_cfg = {
    "name": "ResNet50",
    "num_classes": 1000,
    "replace_with_dilation": [True, True, False],
    "use_feature_only": [2],
    "pretrain": "./ResNet50.pth"
}

c.RetinaNet.reg_sub_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "ReLU",
    },
    "initializer": {
        "name": "StandardInitializer"
    }
}

c.RetinaNet.cls_sub_cfg = {
    "norm": {
        "name": "Identity",
    },
    "nonlinear": {
        "name": "ReLU",
    },
    "initializer": {
        "name": "StandardInitializer"
    }
}

