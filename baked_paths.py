from pathlib import Path

DATA_DIRNAME = 'data'
QUERIES_FILENAME = 'queries.csv'
CURRICULUM_LIST_FILENAME = 'curriculums.txt'
CURRICULUMS_DIRNAME = 'currics'
CURRICULUM_TABLE_FILENAME = 'table.csv'

CWD = Path.cwd()
DATA_PATH = CWD / DATA_DIRNAME
STORED_QUERIES_PATH = DATA_PATH / QUERIES_FILENAME
CURRICULUM_LIST_PATH = DATA_PATH / CURRICULUM_LIST_FILENAME
CURRICULUM_DIR_PATH = DATA_PATH / CURRICULUMS_DIRNAME
