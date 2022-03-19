import math
from shutil import move
from datetime import datetime

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer

from dataloader import get_dataloader
from model import LeNet5

config = {"BATCH_SIZE": 128, "NUM_EPOCHS": 5}
colossalai.launch(config=config, rank=0, world_size=1, host="127.0.0.1", port=1234)


def train(model, optimizer):
    train_dataloader, test_dataloader = get_dataloader(gpc)

    logger = get_dist_logger()
    criterion = torch.nn.CrossEntropyLoss()

    # exponentially increase learning rate from low to high
    def lrs(batch):
        low = math.log2(1e-4)
        high = math.log2(10)
        return 2 ** (
            low + (high - low) * batch / len(train_dataloader) / gpc.config.NUM_EPOCHS
        )

    # lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrs)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
    )
    # build a timer to measure time
    timer = MultiTimer()

    # create a trainer object
    trainer = Trainer(engine=engine, timer=timer, logger=logger)

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),
        # you can uncomment these lines if you wish to use them
        hooks.TensorboardHook(log_dir="./tb_logs", ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_dataloader=test_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True,
    )

    time = datetime.now().strftime("%H%M%S")
    opt = str(optimizer).split(" ")[0]
    move("tb_logs/ParallelMode.GLOBAL_rank_0", f"tb_logs/exp{time}-{opt}")


def test_SGD(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    train(model, optimizer)


def test_Adam(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer)


def test_AdamW(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, optimizer)


def test_RAdam(lr):
    model = LeNet5(n_classes=10)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    train(model, optimizer)


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
