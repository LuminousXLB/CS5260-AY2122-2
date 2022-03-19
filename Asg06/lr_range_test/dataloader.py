from pathlib import Path

from torchvision import transforms
from torchvision.datasets import MNIST
from colossalai.context import ParallelContext
from colossalai.utils import get_dataloader as _get

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


def get_dataloader(ctx: ParallelContext):

    train_dataloader = _get(
        dataset=train_dataset,
        shuffle=True,
        batch_size=ctx.config.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
    )

    test_dataloader = _get(
        dataset=test_dataset,
        add_sampler=False,
        batch_size=ctx.config.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
