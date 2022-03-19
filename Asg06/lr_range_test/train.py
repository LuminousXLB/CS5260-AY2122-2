from datetime import datetime
from shutil import move

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer


def train(
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader, prefix="exp"
):
    logger = get_dist_logger()
    criterion = torch.nn.CrossEntropyLoss()

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
        hooks.AccuracyHook(accuracy_func=Accuracy()),
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
    move("tb_logs/ParallelMode.GLOBAL_rank_0", f"tb_logs/{prefix}{time}-{opt}")
