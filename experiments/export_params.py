#! /usr/bin/env python3

"""
Usage: export_params.py [path-to-training-checkpoint] [output-path]

The model definitions in the experiments and the optimized folder differ, such that the training checkpoints generated
by the experiments can not directly be used to initialize GBaRD from the optimized folder. This script translates
between both model definitions. Pass the path of the training-checkpoint that should be converted as the first parameter
and the output path as the second. If successful, a new folder will be created at the specified output path such that
GBaRD can be initialized via
GBaRD.from_pretrained("[output-path]")
Note that only WS3 1x1x2 models can be converted since GBaRD only implements the WS3 1x1x2 variants.
"""

import sys
import logging
from torch import load
from typing import Any
from pathlib import Path

sys.path.append(str(Path("../optimized/").absolute()))
from gbard import GBaRD, GBaRDConfig

assert len(sys.argv) == 3, "Excactly 2 arguments are expected"
logging.info(f"Loading pre-trained model from training checkpoint at {sys.argv[1]}")
trained = load(sys.argv[1])

assert isinstance(trained, dict), f"{sys.argv[1]} does not seem to be a training checkpoint"
assert "state_dict" in trained, "The checkpoint does not contain a pretrained model"

sdict: dict[str, Any] = trained["state_dict"]
if any(key.startswith("student.") for key in sdict.keys()):
    logging.info("The checkpoint seems to contain a student model")
    prefix = "student."
else:
    prefix = ""

logging.info("Translating the state dictionary")
newdict = {}
for key, value in sdict.items():
    if not key.startswith(prefix):
        continue
    if key.startswith(f"{prefix}doc_encoder.initial.") or key.startswith(f"{prefix}doc_encoder.readout."):
        suffix = key[len(f"{prefix}doc_encoder."):]
        newdict[f"graphembedding.{suffix}"] = value
    elif key.startswith(f"{prefix}doc_encoder.heads.0.gcns.0."):
        suffix = key[len(f"{prefix}doc_encoder.heads.0.gcns.0."):]
        newdict[f"graphembedding.gcn1.{suffix}"] = value
    elif key.startswith(f"{prefix}doc_encoder.heads.0.gcns.1."):
        suffix = key[len(f"{prefix}doc_encoder.heads.0.gcns.1."):]
        newdict[f"graphembedding.gcn2.{suffix}"] = value
    elif key.startswith(f"{prefix}doc_encoder.heads.0."):
        suffix = key[len(f"{prefix}doc_encoder.heads.0."):]
        newdict[f"graphembedding.{suffix}"] = value
    elif key.startswith(f"{prefix}colbert."):
        suffix = key[len(f"{prefix}colbert."):]
        newdict[f"colbert.{suffix}"] = value
    elif key.startswith(f"{prefix}hidden_loss."):
        # Ignore
        pass
    else:
        raise KeyError(key)


logging.info("Initializing GBaRD")
gbard = GBaRD(GBaRDConfig())
# Note that "construction.embeddings" are not set in newdict. These will be loaded from DistilBERT
gbard.load_state_dict(newdict, strict=False)

logging.info(f"Saving pre-trained model to {sys.argv[2]}")
gbard.save_pretrained(sys.argv[2])