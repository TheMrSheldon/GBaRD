#! /usr/bin/env python3

"""
Usage:
./evaluate_trecdl19doc.py
"""

from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from gbard import GBaRD

from utils import TRECDL2019Document

# Download and initialize TREC DL 2019 Doc and Passage
download_dir = Path("./downloaded_datasets")

trecdoc = TRECDL2019Document(download_dir / "trecdl19doc")
trecdocdl = DataLoader(
    trecdoc,
    batch_size=50,
    collate_fn=trecdoc.collate_fn,
    worker_init_fn=trecdoc.worker_init_fn,
)

# Initialize GBaRD
gbard = GBaRD.from_pretrained("pretrained/gbard-ws3_1x1x2-mmse-8mask")

# Evaluate on TREC DL 2019 Document
result = defaultdict(dict[str, float])
for batch in tqdm(trecdocdl):
    queries, docs, qids, dids = batch
    grades = gbard(queries, docs)
    for qid, did, grade in zip(qids, dids, grades):
        result[qid][did] = float(grade)

print(f"Evaluation results on TREC DL'19 Document:\n{trecdoc.evaluate(dict(result))}")
