import math
from pathlib import Path

import colossalai
import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
from torchvision.datasets import MNIST


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs


config = {"BATCH_SIZE": 128, "NUM_EPOCHS": 5}

colossalai.launch(config=config, rank=0, world_size=1, host="127.0.0.1", port=1234)

logger = get_dist_logger()

# build

model = LeNet5(n_classes=10)

# build dataloaders
train_dataset = MNIST(
    root=Path("./tmp/"),
    download=True,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)

test_dataset = MNIST(
    root=Path("./tmp/"),
    train=False,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)

train_dataloader = get_dataloader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=gpc.config.BATCH_SIZE,
    num_workers=1,
    pin_memory=True,
)

test_dataloader = get_dataloader(
    dataset=test_dataset,
    add_sampler=False,
    batch_size=gpc.config.BATCH_SIZE,
    num_workers=1,
    pin_memory=True,
)

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
