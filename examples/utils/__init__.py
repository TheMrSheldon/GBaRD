import gzip
import math
import shutil
import tarfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from pandas import DataFrame, read_csv
from pytrec_eval import RelevanceEvaluator, parse_qrel
from torch.utils.data import IterableDataset, get_worker_info


class _TRECDL19Dataset(IterableDataset):
    def __init__(self, path: Path, qrels_file: Path) -> None:
        path.mkdir(exist_ok=True)
        self.path = path
        self.qrels_file = qrels_file

    def download(self, url: str, ignore_existing: bool = False) -> bool:
        filename = Path(urlparse(url).path).name
        print(f"Downloading {filename}", end="", flush=True)
        target_file = self.path / filename
        if ignore_existing or not target_file.exists():
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                print("    [FAILED]", flush=True)
                return False
            with target_file.open("wb") as f:
                for chunk in response.iter_content(chunk_size=2**27):
                    f.write(chunk)
        print("    [DONE]", flush=True)
        return True

    def extract(self, archive: str, archive_entry: str, ignore_existing: bool = False) -> bool:
        raise NotImplementedError

    def worker_init_fn(self, worker_id):
        worker_info = get_worker_info()
        per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
        firstIdx = worker_id * per_worker
        lastIdx = min(firstIdx + per_worker, len(self))
        self.qrels = self.qrels.head(lastIdx).tail(lastIdx - firstIdx)

    def collate_fn(self, xs):
        return list(zip(*xs))

    def __len__(self) -> int:
        return len(self.qrels)

    def __iter__(self) -> Iterable[tuple[str, str, str, str]]:
        for qid, _, did, rel in self.qrels.itertuples(index=False):
            query = self.queries.loc[qid]
            doc = self.docs.loc[did]
            yield str(query["content"]), str(doc["content"]), str(qid), str(did)

    def get_qrels(self) -> dict[str, dict[str, int]]:
        with Path(self.qrels_file).open("r") as f:
            return parse_qrel(f)

    def get_relevance_level(self) -> int:
        return 1

    def evaluate(
        self,
        results: dict[str, dict[str, float]],
        metrics: list[str] = ["recip_rank", "map", "ndcg_cut.10"],
    ) -> dict[str, float]:
        qrels = self.get_qrels()
        relevance_level = self.get_relevance_level()

        evaluator = RelevanceEvaluator(qrels, metrics, relevance_level=relevance_level)
        eval = evaluator.evaluate(results)
        df = DataFrame([*eval.values()])
        return df.mean().to_dict()


class TRECDL2019Document(_TRECDL19Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__(path, path / "2019qrels-docs.txt")
        assert self.download("https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz")
        assert self.download("https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz")
        assert self.download("https://trec.nist.gov/data/deep/2019qrels-docs.txt")
        assert self.extract("msmarco-docs.tsv.gz", "msmarco-docs.tsv")
        assert self.extract("msmarco-test2019-queries.tsv.gz", "msmarco-test2019-queries.tsv")

        self.qrels: DataFrame = read_csv(
            self.path / "2019qrels-docs.txt",
            sep=" ",
            names=["q_id", "unused", "doc_id", "rel"],
            header=None,
        )
        self.docs: DataFrame = read_csv(
            self.path / "msmarco-docs.tsv",
            sep="\t",
            names=["doc_id", "url", "title", "content"],
            header=None,
            index_col="doc_id",
        )
        self.queries: DataFrame = read_csv(
            self.path / "msmarco-test2019-queries.tsv",
            sep="\t",
            names=["q_id", "content"],
            header=None,
            index_col="q_id",
        )

    def extract(self, archive: str, archive_entry: str, ignore_existing: bool = False) -> bool:
        print(f"Extracting {archive_entry}", end="", flush=True)
        target_file = self.path / archive_entry
        if ignore_existing or not target_file.exists():
            try:
                with gzip.open(self.path / archive, mode="rb") as archivefile:
                    with target_file.open(mode="bw") as f:
                        shutil.copyfileobj(archivefile, f)
            except Exception:
                print("    [FAILED]", flush=True)
                return False
        print("    [DONE]", flush=True)
        return True


class TRECDL2019Passage(_TRECDL19Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__(path, path / "2019qrels-pass.txt")
        assert self.download("https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz")
        assert self.download("https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz")
        assert self.download("https://trec.nist.gov/data/deep/2019qrels-pass.txt")
        assert self.extract("collection.tar.gz", "collection.tsv")
        assert self.extract("queries.tar.gz", "queries.eval.tsv")

        self.qrels: DataFrame = read_csv(
            self.path / "2019qrels-pass.txt",
            sep=" ",
            names=["q_id", "unused", "doc_id", "rel"],
            header=None,
        )
        self.docs: DataFrame = read_csv(
            self.path / "collection.tsv",
            sep="\t",
            names=["doc_id", "content"],
            header=None,
            index_col="doc_id",
        )
        self.queries: DataFrame = read_csv(
            self.path / "queries.eval.tsv",
            sep="\t",
            names=["q_id", "content"],
            header=None,
            index_col="q_id",
        )

    def extract(self, archive: str, archive_entry: str, ignore_existing: bool = False) -> bool:
        print(f"Extracting {archive_entry}", end="", flush=True)
        target_file = self.path / archive_entry
        if ignore_existing or not target_file.exists():
            try:
                with tarfile.open(self.path / archive, mode="r") as archivefile:
                    with archivefile.extractfile(archive_entry) as entry:
                        with target_file.open(mode="bw") as f:
                            shutil.copyfileobj(entry, f)
            except Exception:
                print("    [FAILED]", flush=True)
                return False
        print("    [DONE]", flush=True)
        return True

    def get_relevance_level(self) -> int:
        return 2
