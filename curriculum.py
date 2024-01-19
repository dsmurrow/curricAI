from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance

from ai_calls import embed_string
from baked_paths import CURRICULUM_DIR_PATH, CURRICULUM_TABLE_FILENAME, CURRICULUM_MAPPING_FILENAME

class Mapping:
    def __init__(self, path: Path):
        self._path = path

        empty_index = pd.Index([], name='Name')

        if not path.exists():
            path.touch()
            self._df = pd.DataFrame(columns=['Description', 'Standard', 'Standard Description'], index=empty_index)
        else:
            try:
                self._df = pd.read_csv(path, index_col='Name')
            except pd.errors.EmptyDataError:
                self._df = pd.DataFrame(columns=['Description', 'Standard', 'Standard Description'], index=empty_index)

    def add_mapping(self, qname, qdesc, std, std_desc):
        entry_dict = {'Description': [qdesc], 'Standard': [std], 'Standard Description': [std_desc]}

        idx = pd.Index([qname], name='Name')

        df = pd.DataFrame(entry_dict, index=idx)

        self._df = pd.concat([self._df, df])

    def remove(self, name: str):
        self._df.drop(name, inplace=True)

    def save(self):
        self._df.to_csv(self.path)

    @property
    def names(self) -> list[str]:
        return self._df.index.tolist()

    @property
    def standards(self) -> list[str]:
        return self._df.Standard.tolist()

    @property
    def path(self) -> Path:
        return self._path

    def __getitem__(self, index: str | int) -> pd.Series:
        if isinstance(index, str):
            return self._df.loc[index]
        else:
            return self._df.iloc[index]


class Curriculum:
    _DF_EMBEDDING_COLUMN_NAME = 'embedding'

    def __init__(self, name: str, df: pd.DataFrame = None, *, table_path: Path = None, mapping_path: Path = None):
        self._name = name

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

        if mapping_path is None:
            mapping_path = self.table_path.parent / CURRICULUM_MAPPING_FILENAME

        self._mapping = Mapping(mapping_path)

    def query(self, query: str | list[float] | np.ndarray, include_similarity=False) -> pd.DataFrame:
        if not isinstance(query, str):
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
        self._df[Curriculum._DF_EMBEDDING_COLUMN_NAME] = self._df[Curriculum._DF_EMBEDDING_COLUMN_NAME].apply(list)
        self._df.to_csv(self._table_path)
        self._df[Curriculum._DF_EMBEDDING_COLUMN_NAME] = self._df[Curriculum._DF_EMBEDDING_COLUMN_NAME].apply(np.array)

    @property
    def name(self) -> str:
        return self._name

    @property
    def mapping(self) -> Mapping:
        return self._mapping

    @property
    def table_path(self) -> Path:
        return self._table_path

    @table_path.setter
    def table_path(self, new_path: Path | str):
        self._table_path = new_path

    @property
    def directory(self) -> Path:
        return self._table_path

    def __getitem__(self, index: str):
        return self._df.loc[index]
