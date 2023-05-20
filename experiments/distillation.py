#! /usr/bin/env python3

"""
Usage: distillation.py [hydra-conf-parameters...]

Trains the specified student variant using our proposed distillation loss. `hydra-conf-parameters` are
[hydra](https://hydra.cc/) parameters belonging to the config file in `hydra_conf/distillation.yaml`.

Example: The following creates a run called `[name]` that is trained on the 2 GPUs with IDs 0 and 3. For the dataset,
TREC DL'19 Passage is used, which is located at `[path-to-dataset]` (if it does not exist there, it is downloaded).
Training is run for 20 epochs. Per default, a `checkpoint` folder is created, in which all intermediary checkpoints are
stored. Training can additionally be tracked via the tensorboard files written to `lightning_logs`.
```bash
./distillation.py \
	run_name=[name] \
	used_gpus=[0,3] \
	trainer.devices=2 \
	trainer.accelerator=gpu \
    datamodule=trec19pass \
	datamodule.data_dir="[path-to-dataset]" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=10 \
	trainer.limit_val_batches=0 \
	trainer.max_epochs=20 \
	trainer.strategy.find_unused_parameters=True
```
"""

from pathlib import Path

import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ranking_utils.model.data import DataProcessor

import common


@hydra.main(config_path="hydra_conf", config_name="distillation", version_base=None)
def main(config: DictConfig):
    print("Setting distillation up", flush=True)
    seed_everything(config.seed)
    common.set_cuda_devices_env(config.used_gpus)

    print("Instantiating trainer", flush=True)
    checkpointcb = ModelCheckpoint(dirpath=config.checkpoint_path, save_top_k=-1, filename="{epoch:02d}")
    earlystopping = EarlyStopping(monitor="val_loss", mode="min")
    trainer = hydra_inst(config.trainer, callbacks=[checkpointcb, earlystopping])
    assert isinstance(trainer, Trainer)

    print("Fetching cache paths", flush=True)
    keys = {"precision": trainer.precision}
    cache_root = Path(config.cache_root)
    model_cache = cache_root / config.model_cache.format(**keys) if config.model_cache else None
    processor_cache = cache_root / config.processor_cache.format(**keys) if config.processor_cache else None
    print(f"Model cache: {model_cache}")
    print(f"Processor cache: {processor_cache}")

    print("Instantiating model", flush=True)
    model = hydra_inst(config.ranker.model, cache_dir=model_cache)
    data_processor = hydra_inst(config.ranker.data_processor, cache_dir=processor_cache)
    assert isinstance(model, LightningModule)
    assert isinstance(data_processor, DataProcessor)
    print(f"Running new test using:\n{model.hparams}", flush=True)

    print("Instantiating datamodule", flush=True)
    datamodule = hydra_inst(config.datamodule, data_processor=data_processor)
    assert isinstance(datamodule, LightningDataModule)
    print(data_processor.graph_construction)

    checkpoint = None
    if config.checkpoint is not None:
        checkpoint = Path(config.checkpoint_path) / config.checkpoint
        print(f"Resuming from checkpoint at {checkpoint}")
    print("Training", flush=True)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
