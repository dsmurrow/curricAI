from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance

from ai_calls import embed_string
from baked_paths import CURRICULUM_DIR_PATH, CURRICULUM_TABLE_FILENAME


class Curriculum:
    _DF_EMBEDDING_COLUMN_NAME = 'embedding'

    def __init__(self, name: str, df: pd.DataFrame = None, *, table_path: Path = None, save_on_del=False):
        self._name = name
        self.save_on_del = save_on_del

        self.table_path = table_path or CURRICULUM_DIR_PATH / \
            name / CURRICULUM_TABLE_FILENAME

        if df is None:
            self._df = pd.read_csv(self.table_path, index_col='Standard')
            self._df[Curriculum._DF_EMBEDDING_COLUMN_NAME] = self._df.embedding.apply(
                literal_eval).apply(np.array)
        else:
            self._df = df.copy(deep=True)

        if Curriculum._DF_EMBEDDING_COLUMN_NAME not in self._df.columns:
            self._df[Curriculum._DF_EMBEDDING_COLUMN_NAME] = self._df.Description.apply(
                embed_string)

    def query(self, query: str | list[float] | np.ndarray, include_similarity=False) -> pd.DataFrame:
        if isinstance(query, str):
            query_embedding = query
        else:
            query_embedding = embed_string(query)

        df = self._df.copy()
        df['similarity'] = df.embedding.apply(
            lambda x: distance.cosine(x, query_embedding))
        df.sort_values('similarity', inplace=True)
        if include_similarity:
            return df[["Description", "similarity"]]
        else:
            return df[["Description"]]

    def save(self):
        self._df.to_csv(self._table_path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def table_path(self) -> Path:
        return self._table_path

    @table_path.setter
    def table_path(self, new_path: Path | str):
        self._table_path = new_path

    @property
    def directory(self) -> Path:
        return self._table_path.absolute()

    @property
    def save_on_del(self) -> bool:
        return self._save_on_del

    @save_on_del.setter
    def save_on_del(self, will_save: bool):
        self._save_on_del = will_save

    def __getitem__(self, index: str):
        return self._df[index]

    def __del__(self):
        if self._save_on_del:
            self.save()
