from itertools import zip_longest
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re
import tiktoken

client = OpenAI()

encoding = tiktoken.get_encoding('cl100k_base')

cwd = Path.cwd()
data_path = cwd / 'data'
curriculum_table_path = data_path / 'curriculums.txt'
curriculum_path = data_path / 'currics'

max_tokens = 8000

def clear():
	name = os.name

	if name == 'nt':
		os.system('cls')
	else:
		os.system('clear')

def intify(x):
	try:
		return int(x)
	except:
		return -1

def read_sheet(path):
	extension = path.suffix

	if extension == '.csv':
		return pd.read_csv(path)
	elif extension in {'.xlsx', '.ods'}:
		return pd.read_excel(path, sheet_name=None)
	else:
		return None

def print_list_and_query_input(header, items):
	max_accepted_input = len(items)
	is_valid = lambda x: x >= 1 and x <= max_accepted_input

	user_input = -1
	while not is_valid(user_input):
		clear()
		print(header)
		for (i, item) in enumerate(items):
			print(f'{i+1}. {item}')

		user_input = intify(input('Choose an option: '))

	return user_input

def scan_new_curriculums(path):
	has_right_columns = lambda df: 'Description' in df.columns and 'Standard' in df.columns
	sanitize = lambda df: df[['Standard', 'Description']].dropna()

	files = filter(lambda p: p.is_file(), path.iterdir())

	dfs = []
	for child in files:
		maybe_df = read_sheet(child)
		if isinstance(maybe_df, dict):
			for (name, df) in maybe_df.items():
				if has_right_columns(df):
					dfs.append((name, sanitize(df)))
		elif maybe_df is None:
			continue
		elif has_right_columns(maybe_df):
			dfs.append((child.name, sanitize(maybe_df)))

	return dfs		

def embed_string(string, model='text-embedding-ada-002'):
	text = text.replace('\n', ' ')
	return client.embeddings.create(input=text, model=model)
	

def establish_new_curriculums(named_dfs):
	token_len = lambda x: len(encoding.encode(x))
	too_many_tokens = lambda df: df.Description.apply(token_len).gt(max_tokens).any()
	filtered_dfs = filter(lambda p: not too_many_tokens(p[1]), named_dfs)


	for (name, df) in filtered_dfs: # TODO: Check for duplicates
		header = (
			f'New curriculum "{name}"\n'
			'Would you like to give it a different name?'
		)	
		options = ['Yes', 'No']

		selection = print_list_and_query_input(header, options)

		if selection == 1:
			name = input('New name: ')
			name = re.sub('[#%&{}\\<>*?/$!\'":@+`|=]', '', name)

		# TODO: Make directories for them


		# TODO: Get embeddings
		

def scan_for_curriculums():
	if not curriculum_table_path.exists() or not curriculum_table_path.is_file():
		curriculum_table_path.touch()

	currics = []

	with open(curriculum_table_path, 'r') as file:
		lines = map(lambda x: x[:-1], file.readlines())

		# Chunk the elements by pairs
		args = [lines] * 2
		pairs = zip_longest(*args, fillvalue=None)

		# TODO: put into list
					
	return currics
		
def main_loop():
	SCAN_OPTION_STRING = 'Scan'
	EXIT_OPTION_STRING = 'Exit'
	items = ['bruh', 'two', 'three', SCAN_OPTION_STRING, EXIT_OPTION_STRING]

	header = 'Chews!'

	selection = 1
	while items[selection - 1] != EXIT_OPTION_STRING:
		curriculums = scan_for_curriculums()
		selection = print_list_and_query_input(header, items)
		
		if items[selection - 1] == SCAN_OPTION_STRING:
			currics = scan_new_curriculums(cwd)
			establish_new_curriculums(currics)

if __name__ == '__main__':
	if not data_path.exists() or not data_path.is_dir():
		data_path.mkdir()
	if not curriculum_path.exists() or not curriculum_path.is_dir():
		curriculum_path.mkdir()

	main_loop()
	
