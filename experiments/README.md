# Experiments
This folder contains the code we used to come up with and test our model architecture. It is **not** efficient but designed for flexibility. We upload this code for better reproducibility or our thesis. As such, there have not been any functional changes to it since we ran our experiments.

The following are all the scripts for training and evaluating our model. Before running them, `cd` into this folder and install the requirements from `requirements.txt` as well as `pytorch`. Expand each section for more information on commandline arguments.


<details>
<summary><code>benchmark.py</code> &ndash; Benchmarking MMSE-ColBERT's and WS3 1&times;1&times;2's document encoder</summary>
Usage:

```bash
benchmark.py [path-to-dataset] [Proposal | MMSEColBERT]
```

Benchmarks the specified model on TREC DL'19 Passage. If no model was specified (i.e., neither `Proposal` nor `MMSEcolBERT`
was passed as an argument), both models are benchmarked. `path-to-dataset` denotes the path the dataset is located at.
If it does not exist, the path will be created and the dataset will be downloaded.

The benchmark is only performed on the document encoder since the query encoder and interaction function of both models
are identical.
</details>

<details>
<summary><code>export_params.py</code> &ndash; Export parameters learnt for WS3 1&times;1&times;2 to be used with GBaRD (see the `optimized` folder)</summary>
Usage:

```bash
export_params.py [path-to-training-checkpoint] [output-path]
```

The model definitions in the experiments and the optimized folder differ, such that the training checkpoints generated
by the experiments can not directly be used to initialize GBaRD from the optimized folder. This script translates
between both model definitions. Pass the path of the training-checkpoint that should be converted as the first parameter
and the output path as the second. If successful, a new folder will be created at the specified output path such that
GBaRD can be initialized via

```python
GBaRD.from_pretrained("[output-path]")
```

Note that only WS3 1&times;1&times;2 models can be converted since GBaRD only implements the WS3 1&times;1&times;2 variants.
</details>


<details>
<summary><code>distillation.py</code> &ndash; Distilling MMSE-ColBERT into WS3 1&times;1&times;2 using a variation of TinyBERT's distillation loss</summary>
Usage:

```bash
distillation.py [hydra-conf-parameters...]
```

Trains the specified student variant using our proposed distillation loss. `hydra-conf-parameters` are [hydra](https://hydra.cc/) parameters belonging to the config file in `hydra_conf/distillation.yaml`.

**Example:** The following creates a run called `[name]` that is trained on the 2 GPUs with IDs 0 and 3. For the dataset,
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
</details>


<details>
<summary><code>mmsedistill.py</code> &ndash; Distilling MMSE-ColBERT into WS3 1&times;1&times;2 using a Margin-MSE loss</summary>
Usage:

```bash
mmsedistill.py [hydra-conf-parameters...]
```

Trains the specified student variant using Margin-MSE loss. `hydra-conf-parameters` are [hydra](https://hydra.cc/) parameters belonging to the config file in `hydra_conf/distillation.yaml`.

**Example:** The following creates a run called `[name]` that is trained on the 2 GPUs with IDs 0 and 3. For the dataset,
TREC DL'19 Passage is used, which is located at `[path-to-dataset]` (if it does not exist there, it is downloaded).
Training is run for 20 epochs. Per default, a `checkpoint` folder is created, in which all intermediary checkpoints are
stored. Training can additionally be tracked via the tensorboard files written to `lightning_logs`.

```bash
./mmsedistill.py \
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
</details>


<details>
<summary><code>trec_eval.py</code> &ndash; Evaluating a trained model on a TREC DL'19 dataset</summary>
Usage:

```bash
trec_eval.py [hydra-conf-parameters...]
```

Evaluates a previously trained model using the provided parameters.

Example: The following evalutes run called `[name]` on TREC DL '19 passage, which is located at `[path-to-dataset]` (if
it does not exist there, it is downloaded). For the ranking definition the hydra configuration [ranker] is used and
evaluation is performed on the checkpoint loaded from [checkpoint-name].
Evaluation is performed on a single GPU.

```bash
./trec_eval.py \
    ranker=[ranker] \
    run_name="[name]" \
    checkpoint="[checkpoint-name]" \
    processor_cache=null \
    model_cache=null \
    trainer.accelerator=gpu \
    trainer.precision=32 \
    datamodule=trec19pass \
    datamodule.data_dir="[path-to-dataset]" \
    datamodule.num_workers=12 \
```
</details>


<details>
<summary><code>trec_eval_mmse_distill.py</code> &ndash; Evaluating a MMSE-distilled model on a TREC DL'19 dataset</summary>
Usage:

```bash
trec_eval_mmse_distill.py [hydra-conf-parameters...]
```

Evaluates the student model of a run of Margin-MSE distillation using the provided parameters.

Example: The following evalutes run called `[name]` on TREC DL '19 passage, which is located at `[path-to-dataset]` (if
it does not exist there, it is downloaded). For the ranking definition the hydra configuration [ranker] is used and
evaluation is performed on the checkpoint loaded from [checkpoint-name].
Evaluation is performed on a single GPU.
```bash
./trec_eval_mmse_distill.py \
    ranker=[ranker] \
	run_name="[name]" \
	checkpoint="[checkpoint-name]" \
	processor_cache=null \
	model_cache=null \
	trainer.accelerator=gpu \
	trainer.precision=32 \
	datamodule=trec19pass \
	datamodule.data_dir="[path-to-dataset]" \
	datamodule.num_workers=12 \
```
</details>
