import io
import shutil
import tarfile
from pathlib import Path
from typing import Iterator, Optional, Union
from urllib.parse import urlparse

import pandas
import requests
from pytorch_lightning import LightningDataModule
from ranking_utils.model.data import subset_dataset
from ranking_utils.model.data.h5 import (
    ContrastiveTrainingInstance,
    DataProcessor,
    PairwiseTrainingInstance,
    PointwiseTrainingInstance,
    PredictionDataset,
    PredictionInstance,
    TrainingDataset,
    TrainingMode,
    ValTestDataset,
    ValTestInstance,
)
from torch.utils.data import DataLoader

from ..trec_eval import load_qrels_from_file


class TREC2019PassageTrain(TrainingDataset):
    def __init__(self, data_processor: DataProcessor, root_dir: Path) -> None:
        super().__init__(data_processor, TrainingMode.POINTWISE)
        self.root_dir = root_dir
        self.qrels: pandas.DataFrame = pandas.read_csv(
            root_dir / "qrels.train.tsv", sep="\t", names=["q_id", "unused", "doc_id", "rel"], header=None
        )
        self.docs: pandas.DataFrame = pandas.read_csv(
            root_dir / "collection.tsv", sep="\t", names=["doc_id", "content"], header=None, index_col="doc_id"
        )
        self.queries: pandas.DataFrame = pandas.read_csv(
            root_dir / "queries.train.tsv", sep="\t", names=["q_id", "content"], header=None, index_col="q_id"
        )

    def _num_pointwise_instances(self) -> int:
        """Return the number of pointwise training instances.
        Returns:
            int: The number of pointwise training instances.
        """
        return len(self.qrels)

    def _num_pairwise_instances(self) -> int:
        raise io.UnsupportedOperation

    def _num_contrastive_instances(self) -> int:
        raise io.UnsupportedOperation

    def _get_pointwise_instance(self, index: int) -> PointwiseTrainingInstance:
        """Return the pointwise training instance corresponding to an index.
        Args:
            index (int): The index.
        Returns:
            PointwiseTrainingInstance: The corresponding training instance.
        """
        row = self.qrels.iloc[index]
        query = self.queries.loc[row["q_id"]]
        doc = self.docs.loc[row["doc_id"]]
        return str(query["content"]), str(doc["content"]), int(row["rel"])

    def _get_pairwise_instance(self, index: int) -> PairwiseTrainingInstance:
        raise io.UnsupportedOperation

    def _get_contrastive_instance(self, index: int) -> ContrastiveTrainingInstance:
        raise io.UnsupportedOperation


class TREC2019PassageTest(ValTestDataset):
    def __init__(
        self, data_processor: DataProcessor, root_dir: Path, qrels_file: str = "2019qrels-pass.txt", sep=" "
    ) -> None:
        super().__init__(data_processor)
        self.root_dir = root_dir
        self.qrels: pandas.DataFrame = pandas.read_csv(
            root_dir / qrels_file, sep=sep, names=["q_id", "unused", "doc_id", "rel"], header=None
        )
        self.docs: pandas.DataFrame = pandas.read_csv(
            root_dir / "collection.tsv", sep="\t", names=["doc_id", "content"], header=None, index_col="doc_id"
        )
        self.queries: pandas.DataFrame = pandas.read_csv(
            root_dir / "queries.dev.tsv", sep="\t", names=["q_id", "content"], header=None, index_col="q_id"
        )

    def _num_instances(self) -> int:
        """Return the number of instances.

        Returns:
            int: The number of instances.
        """
        return len(self.qrels)

    def _get_instance(self, index: int) -> ValTestInstance:
        """Return the instance corresponding to an index.

        Args:
            index (int): The index.

        Returns:
            ValTestInstance: The corresponding instance.
        """
        row = self.qrels.iloc[index]
        query = self.queries.loc[row["q_id"]]
        doc = self.docs.loc[row["doc_id"]]
        # Return relevance-1 since 0, 1 are considered irrelevant whereas 2, 3 are considered relevant
        return str(query["content"]), str(doc["content"]), (int(row["q_id"])), (int(row["rel"]) - 1)


class TREC2019PassagePredict(PredictionDataset):
    def __init__(
        self,
        data_processor: DataProcessor,
        root_dir: Path,
        qrels_file: str = "2019qrels-pass.txt",
        sep=" ",
        queries_file: str = "queries.eval.tsv",
    ) -> None:
        super().__init__(data_processor)
        self.root_dir = root_dir
        self.qrels: pandas.DataFrame = pandas.read_csv(
            root_dir / qrels_file, sep=sep, names=["q_id", "unused", "doc_id", "rel"], header=None
        )
        self.docs: pandas.DataFrame = pandas.read_csv(
            root_dir / "collection.tsv", sep="\t", names=["doc_id", "content"], header=None, index_col="doc_id"
        )
        self.queries: pandas.DataFrame = pandas.read_csv(
            root_dir / queries_file, sep="\t", names=["q_id", "content"], header=None, index_col="q_id"
        )

    def _num_instances(self) -> int:
        """Return the number of instances.

        Returns:
            int: The number of instances.
        """
        return len(self.qrels)

    def _get_instance(self, index: int) -> PredictionInstance:
        """Return the instance corresponding to an index.

        Args:
            index (int): The index.

        Returns:
            ValTestInstance: The corresponding instance.
        """
        row = self.qrels.iloc[index]
        query = self.queries.loc[row["q_id"]]
        doc = self.docs.loc[row["doc_id"]]
        return index, str(query["content"]), str(doc["content"])

    def ids(self) -> Iterator[tuple[int, str, str]]:
        for i in range(self._num_instances()):
            row = self.qrels.iloc[i]
            yield i, str(row["q_id"]), str(row["doc_id"])


class TREC2019Passage(LightningDataModule):
    def __init__(
        self,
        data_processor: DataProcessor,
        data_dir: str,
        batch_size: int,
        num_workers: int = 16,
        limit_train_set: Union[int, float, None] = None,
        limit_test_set: Union[int, float, None] = None,
    ):
        super().__init__()
        self.root_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_processor = data_processor
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.limit_train_set = limit_train_set
        self.limit_test_set = limit_test_set

    def _download(self, url: str, filename: Optional[str] = None, ignore_existing: bool = False) -> bool:
        if filename is None:
            filename = Path(urlparse(url).path).name
        target_file = self.root_dir / filename
        if ignore_existing or not target_file.exists():
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                return False
            with target_file.open("wb") as f:
                for chunk in response.iter_content(chunk_size=2**25):
                    f.write(chunk)
        return True

    def _extract(
        self, archive: str, filename: str, archive_entry: Optional[str] = None, ignore_existing: bool = False
    ) -> bool:
        archive_entry = archive_entry or filename
        target_file = self.root_dir / filename
        if ignore_existing or not target_file.exists():
            try:
                with tarfile.open(self.root_dir / archive, mode="r") as archivefile:
                    with archivefile.extractfile(archive_entry) as entry:
                        with target_file.open(mode="bw") as f:
                            shutil.copyfileobj(entry, f)
                    return True
            except Exception:
                return False
        return True

    def prepare_data(self):
        # Download files (https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html#passage-ranking-dataset)
        # I will not download all files since some are quite big and I have no use for them
        downloads = [
            ("https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",),
            ("https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",),
            ("https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv",),
            ("https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv",),
            # ("https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",),
            # ("https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz",),
            # ("https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.full.tsv.gz",),
            # ("https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz",),
            # ("https://msmarco.blob.core.windows.net/msmarcoranking/top1000.train.tar.gz",),
            ("https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz",),
            # ("https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",),
            # ("https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz",),
            ("https://trec.nist.gov/data/deep/2019qrels-pass.txt",),
        ]
        for download in downloads:
            if not self._download(*download):
                raise Exception(f"Download failed: {download}")
        # This list may not be complete, it only contains those files that I need
        extracts = [
            ("collection.tar.gz", "collection.tsv"),
            ("queries.tar.gz", "queries.dev.tsv"),
            ("queries.tar.gz", "queries.eval.tsv"),
            ("queries.tar.gz", "queries.train.tsv"),
            ("top1000.dev.tar.gz", "top1000.dev"),
        ]
        for extract in extracts:
            if not self._extract(*extract):
                raise Exception(f"Failed to extract file {extract}")

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = TREC2019PassageTrain(self.data_processor, self.root_dir) if stage == "fit" else None
        self.test_dataset = (
            TREC2019PassageTest(self.data_processor, self.root_dir, "2019qrels-pass.txt", sep=" ")
            if stage == "test"
            else None
        )
        self.val_dataset = (
            TREC2019PassageTest(self.data_processor, self.root_dir, "qrels.dev.tsv", sep="\t")
            if stage in ["fit", "validate"]
            else None
        )
        self.predict_dataset = (
            TREC2019PassagePredict(self.data_processor, self.root_dir) if stage == "predict" else None
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=subset_dataset(self.train_dataset, self.limit_train_set),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=subset_dataset(self.test_dataset, self.limit_test_set),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.predict_dataset.collate_fn,
            pin_memory=True,
        )

    def qrels(self) -> dict[str, dict[str, int]]:
        return load_qrels_from_file(self.root_dir / "2019qrels-pass.txt")

    def num_top1000(self) -> int:
        top1000: pandas.DataFrame = pandas.read_csv(
            self.root_dir / "top1000.dev",
            sep="\t",
            names=["q_id", "doc_id", "query", "doc"],
            header=None,
            low_memory=True,
        )
        return len(top1000["query"].unique())

    def top1000(self) -> Iterator[tuple[str, list[str]]]:
        top1000: pandas.DataFrame = pandas.read_csv(
            self.root_dir / "top1000.dev",
            sep="\t",
            names=["q_id", "doc_id", "query", "doc"],
            header=None,
            low_memory=True,
        )
        for query, dataframe in top1000.groupby("query"):
            yield (str(query), [str(doc) for doc in dataframe["doc"]])

    def relevance_level(self) -> int:
        """
        Returns the smallest relevance grade that can be considered relevant.
        For trec '19 passage this is 2 (https://trec.nist.gov/data/deep2019.html).
        """
        return 2

    def teardown(self, stage):
        pass
