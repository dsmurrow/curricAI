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
	files = filter(lambda p: p.is_file(), path.iterdir())

	dfs = []
	for child in files:
		maybe_df = read_sheet(child)
		if isinstance(maybe_df, dict):
			dfs.extend(maybe_df.items())
		elif maybe_df is None:
			continue
		else:
			dfs.append((child.name, maybe_df))

	return dfs		

if __name__ == '__main__':
	if not curriculum_path.exists() or not curriculum_path.is_dir():
		curriculum_path.mkdir()

	items = ['bubkis', 'an option', 'heck yeah', 'chews', 'scan']
	header = 'CHOOSE!!!!'
	selection = print_list_and_query_input(header, items)

	if items[selection - 1] == 'scan':
		currics = scan_new_curriculums(cwd)
		for (name, df) in currics:
			print(name)
