from typing import Optional

from torch_geometric.data import Data


class GraphConstruction:
    def __call__(self, input: str, device: Optional[str] = None) -> Data:
        raise NotImplementedError
