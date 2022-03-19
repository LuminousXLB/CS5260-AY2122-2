import math

import colossalai
import numpy as np
import torch
from colossalai.core import global_context as gpc

from dataloader import get_dataloader
from model import LeNet5
from train import train

config = {"BATCH_SIZE": 128, "NUM_EPOCHS": 5}
colossalai.launch(config=config, rank=0, world_size=1, host="127.0.0.1", port=1234)
train_dataloader, test_dataloader = get_dataloader(gpc)


LR_LIMIT = {
    "SGD": (1.466599e-02, 1.235050e-01),
    "Adam": (2.408037e-05, 5.063813e-04),
    "AdamW": (2.408037e-05, 4.975099e-04),
    "RAdam": (4.092044e-05, 7.126536e-04),
}

STEPS = len(train_dataloader) * gpc.config.NUM_EPOCHS


def MultiStepLR_train(model, optimizer, lr_min, lr_max, segments=5):
    milestones = [int(x) for x in np.linspace(0, 2344, segments)[1:-1]]
    gamma = math.exp((math.log(lr_min) - math.log(lr_max)) / segments)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    train(
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
        prefix="MultiStepLR",
    )


def test_SGD_MultiStepLR():
    lr_min, lr_max = LR_LIMIT["SGD"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr_max, momentum=0.9, weight_decay=5e-4
    )
    MultiStepLR_train(model, optimizer, lr_min, lr_max)


def test_Adam_MultiStepLR():
    lr_min, lr_max = LR_LIMIT["Adam"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
    MultiStepLR_train(model, optimizer, lr_min, lr_max)


def test_AdamW_MultiStepLR():
    lr_min, lr_max = LR_LIMIT["AdamW"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)
    MultiStepLR_train(model, optimizer, lr_min, lr_max)


def test_RAdam_MultiStepLR():
    lr_min, lr_max = LR_LIMIT["RAdam"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr_max)
    MultiStepLR_train(model, optimizer, lr_min, lr_max)


def OneCycleLR_train(model, optimizer, lr_min, lr_max):
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max, total_steps=STEPS
    )
    train(
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
        prefix="OneCycleLR",
    )


def test_SGD_OneCycleLR():
    lr_min, lr_max = LR_LIMIT["SGD"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr_max, momentum=0.9, weight_decay=5e-4
    )
    OneCycleLR_train(model, optimizer, lr_min, lr_max)


def test_Adam_OneCycleLR():
    lr_min, lr_max = LR_LIMIT["Adam"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
    OneCycleLR_train(model, optimizer, lr_min, lr_max)


def test_AdamW_OneCycleLR():
    lr_min, lr_max = LR_LIMIT["AdamW"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)
    OneCycleLR_train(model, optimizer, lr_min, lr_max)


def test_RAdam_OneCycleLR():
    lr_min, lr_max = LR_LIMIT["RAdam"]
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr_max)
    OneCycleLR_train(model, optimizer, lr_min, lr_max)


if __name__ == "__main__":
    import sys
    from time import sleep

    opt = sys.argv[1]

    if opt == "sgd-ms":
        test_SGD_MultiStepLR()
    elif opt == "adam-ms":
        test_Adam_MultiStepLR()
    elif opt == "adamw-ms":
        test_AdamW_MultiStepLR()
    elif opt == "radam-ms":
        test_RAdam_MultiStepLR()
    elif opt == "sgd-oc":
        test_SGD_OneCycleLR()
    elif opt == "adam-oc":
        test_Adam_OneCycleLR()
    elif opt == "adamw-oc":
        test_AdamW_OneCycleLR()
    elif opt == "radam-oc":
        test_RAdam_OneCycleLR()
    else:
        print(opt)
        exit(-1)

    sleep(1)
