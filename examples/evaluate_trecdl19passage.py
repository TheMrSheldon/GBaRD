#! /usr/bin/env python3

"""
Usage:
./evaluate_trecdl19passage.py
"""

from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from gbard import GBaRD

from utils import TRECDL2019Passage

# Download and initialize TREC DL 2019 Doc and Passage
download_dir = Path("./downloaded_datasets")

trecpass = TRECDL2019Passage(download_dir / "trecdl19pass")
trecpassdl = DataLoader(
    trecpass,
    batch_size=50,
    collate_fn=trecpass.collate_fn,
    worker_init_fn=trecpass.worker_init_fn,
)

# Initialize GBaRD
gbard = GBaRD.from_pretrained("pretrained/gbard-ws3_1x1x2-mmse-8mask")

# Evaluate on TREC DL 2019 Passage
result = defaultdict(dict[str, float])
for batch in tqdm(trecpassdl):
    queries, docs, qids, dids = batch
    grades = gbard(queries, docs)
    for qid, did, grade in zip(qids, dids, grades):
        result[qid][did] = float(grade)

print(f"Evaluation results on TREC DL'19 Passage:\n{trecpass.evaluate(dict(result))}")
