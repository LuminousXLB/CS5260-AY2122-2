import math

import colossalai
import torch
from colossalai.core import global_context as gpc

from dataloader import get_dataloader
from model import LeNet5
from train import train

config = {"BATCH_SIZE": 128, "NUM_EPOCHS": 5}
colossalai.launch(config=config, rank=0, world_size=1, host="127.0.0.1", port=1234)
train_dataloader, test_dataloader = get_dataloader(gpc)

# MultiStepLR
# OneCycleLR


def train_wrapper(model, optimizer):
    # exponentially increase learning rate from low to high
    def lrs(batch):
        low = math.log2(1e-4)
        high = math.log2(10)
        return 2 ** (
            low + (high - low) * batch / len(train_dataloader) / gpc.config.NUM_EPOCHS
        )

    # lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrs)

    train(model, optimizer, lr_scheduler, train_dataloader, test_dataloader)


def test_SGD(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    train_wrapper(model, optimizer)


def test_Adam(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_wrapper(model, optimizer)


def test_AdamW(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_wrapper(model, optimizer)


def test_RAdam(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    train_wrapper(model, optimizer)


if __name__ == "__main__":
    import sys

    opt = sys.argv[1]

    if opt == "sgd":
        test_SGD(1)
    elif opt == "adam":
        test_Adam(0.1)
    elif opt == "adamw":
        test_AdamW(0.1)
    elif opt == "radam":
        test_RAdam(0.1)
    else:
        print(opt)
        exit(-1)
