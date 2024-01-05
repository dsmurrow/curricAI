import os
import pandas as pd
from pathlib import Path

cwd = Path.cwd()
curriculum_path = cwd / 'currics'


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
		os.system('clear')
		print(header)
		for (i, item) in enumerate(items):
			print(f'{i+1}. {item}')

		user_input = intify(input('Choose an option: '))

	return user_input

def scan_new_curriculums(path):
	has_right_columns = lambda df: 'Description' in df.columns and 'Standard' in df.columns

	files = filter(lambda p: p.is_file(), path.iterdir())

	dfs = []
	for child in files:
		maybe_df = read_sheet(child)
		if isinstance(maybe_df, dict):
			for (name, df) in maybe_df.items():
				if has_right_columns(df):
					dfs.append((name, df))
		elif maybe_df is None:
			continue
		elif has_right_columns(maybe_df):
			dfs.append((child.name, maybe_df))

	return dfs		

def establish_new_curriculums(named_dfs):
	for (name, df) in named_dfs: # TODO: Check for duplicates
		header = (
			f'New curriculum "{name}".\n'
			'Would you like to give it a different name?'
		)
		options = ['Yes', 'No']

		selection = print_list_and_query_input(header, options)

		if selection == 1:
			name = input('New name: ')

		# TODO: Make directories for them
		df = df[['Standard', 'Description']]

		# TODO: Get embeddings
		
def main_loop():
	SCAN_OPTION_STRING = 'Scan'
	EXIT_OPTION_STRING = 'Exit'
	items = ['bruh', 'two', 'three', SCAN_OPTION_STRING, EXIT_OPTION_STRING]

	header = 'Chews!'

	selection = 1
	while items[selection - 1] != EXIT_OPTION_STRING:
		selection = print_list_and_query_input(header, items)
		
		if items[selection - 1] == SCAN_OPTION_STRING:
			currics = scan_new_curriculums(cwd)
			establish_new_curriculums(currics)

if __name__ == '__main__':
	if not curriculum_path.exists() or not curriculum_path.is_dir():
		curriculum_path.mkdir()

	main_loop()
	
