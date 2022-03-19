import math

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer

from model import LeNet5
from dataloader import get_dataloader

config = {"BATCH_SIZE": 128, "NUM_EPOCHS": 5}

colossalai.launch(config=config, rank=0, world_size=1, host="127.0.0.1", port=1234)

logger = get_dist_logger()

# build
model = LeNet5(n_classes=10)
train_dataloader, test_dataloader = get_dataloader(gpc)

# build criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# exponentially increase learning rate from low to high
def lrs(batch):
    low = math.log2(1e-5)
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
    # hooks.AccuracyHook(accuracy_func=Accuracy()),
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
