from itertools import zip_longest
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re
import shutil
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

def print_list(header, items, indeces=None):
	padding = len(str(len(items)))

	if indeces is None:
		iterator = enumerate(items)
	else:
		iterator = zip(indeces, items)

	print(header)
	for i, item in iterator:
		print(f'{i+1:>{padding}}. {item}')

def print_list_and_query_input(header, items):
	max_accepted_input = len(items)
	is_valid = lambda x: x >= 1 and x <= max_accepted_input

	padding = len(str(len(items)))

	user_input = -1
	while not is_valid(user_input):
		clear()
		print_list(header, items)

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
			name = child.name
			for suffix in child.suffixes:
				name = name.replace(suffix, '')

			dfs.append((name, sanitize(maybe_df)))

	return dfs		

def embed_string(string, model='text-embedding-ada-002'):
	text = text.replace('\n', ' ')
	return client.embeddings.create(input=text, model=model)
	

def establish_new_curriculums(named_dfs, already_used_names=set()):
	SKIP_OPTION_STRING = "I don't want to use this"

	token_len = lambda x: len(encoding.encode(x))
	too_many_tokens = lambda df: df.Description.apply(token_len).gt(max_tokens).any()
	filtered_dfs = filter(lambda p: not too_many_tokens(p[1]), named_dfs)

	options = ['Yes', 'No', SKIP_OPTION_STRING]

	curriculum_table = open(curriculum_table_path, 'a')

	names_used = {name: 1 for name in already_used_names}

	for name, df in filtered_dfs:
		header = (
			f'New curriculum "{name}"\n'
			'Would you like to give it a different name?'
		)	

		selection = print_list_and_query_input(header, options)
	
		selected_option = options[selection - 1]
		if selected_option == SKIP_OPTION_STRING:
			continue
		elif selected_option == 'Yes':
			name = input('New name: ')

			# Get rid of characters that are invalid for filenames
			name = re.sub('[#%&{}\\<>*?/$!\'":@+`|=]', '', name)

		# Disallow duplicate names
		if names_used.setdefault(name, 0) > 0:
			new_name = name + f' ({names_used[name]})'
			while new_name in names_used.keys():
				names_used[name] += 1
				new_name = name + f' ({names_used[name]})'

			names_used[name] += 1
			names_used[new_name] = 0
			
			name = new_name

		names_used[name] += 1

		# Make directories for them
		curriculum_dir = curriculum_path / name
		curriculum_table.write(name + '\n')

		curriculum_dir.mkdir()

		# TODO: Get embeddings
		df['embedding'] = df.Description.apply(len)

		df.to_csv(curriculum_dir / 'table.csv')

	curriculum_table.close()
		

def scan_for_curriculums():
	if not curriculum_table_path.exists() or not curriculum_table_path.is_file():
		curriculum_table_path.touch()

	with open(curriculum_table_path, 'r') as file:
		currics = list(map(lambda x: x[:-1], file))

	return currics

def removing_menu(curriculums):
	header = (
			'Write the numbers corresponding to the curriculums you\'d like to remove.\n'
			'Separate entries by leaving space between them.\n'
			'Type \'*\' to remove all curriculums.\n'
			'Leave blank if you\'d like to leave this menu.'
	)

	clear()
	print_list(header, curriculums)
	selections = input('Selections: ').split()

	if len(selections) == 0:
		return True
	# Delete everything
	elif len(selections) > 0 and selections[0] == '*':
		selections = list(range(1, len(curriculums) + 1))

	index_set = set()
	for item in selections:
		try:
			index_set.add(int(item))
		except:
			input('Invalid input detected. Press Enter to return to main menu.\n')
			return True

	# Input sanitization: In-bounds, 0-based.
	indeces = filter(lambda x: x in range(1, len(curriculums) + 1), index_set)
	indeces = list(map(lambda x: x - 1, indeces))

	clear()
	header = 'Items to be deleted.'
	print_list(header, [curriculums[i] for i in indeces], indeces=indeces)

	selection = input('Delete these items? (Y)es or (N)o: ')
	if selection[0].lower() != 'y':
		return False

	indeces.sort(reverse=True)

	for i in indeces:
		name = curriculums[i]

		curriculum_dir = curriculum_path / name

		shutil.rmtree(curriculum_dir)

		del curriculums[i]

	with open(curriculum_table_path, 'w') as f:
		for curriculum in curriculums:
			f.write(curriculum + '\n')

	return True
		
def main_loop():
	SCAN_OPTION_STRING = 'Scan'
	REMOVE_OPTION_STRING = 'Remove Curriculums'
	EXIT_OPTION_STRING = 'Exit'
	items = [SCAN_OPTION_STRING, REMOVE_OPTION_STRING, EXIT_OPTION_STRING]

	header = 'Chews!'

	current_selection = ''
	while current_selection != EXIT_OPTION_STRING:
		curriculums = scan_for_curriculums()
		
		current_items = curriculums + items

		selection_number = print_list_and_query_input(header, current_items)
		current_selection = current_items[selection_number - 1]

		if current_selection == SCAN_OPTION_STRING:
			currics = scan_new_curriculums(cwd)
			establish_new_curriculums(currics, set(curriculums))
		elif current_selection == REMOVE_OPTION_STRING:
			status = False
			while not status:
				status = removing_menu(curriculums)

if __name__ == '__main__':
	if not data_path.exists() or not data_path.is_dir():
		data_path.mkdir()
	if not curriculum_path.exists() or not curriculum_path.is_dir():
		curriculum_path.mkdir()

	main_loop()
	
