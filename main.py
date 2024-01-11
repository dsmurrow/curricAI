from ast import literal_eval
from enum import Enum
from math import ceil
import numpy as np
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re
from scipy.spatial import distance
import shutil
import tiktoken

UNDER_ALL_HEADERS = '=' * 3

client = OpenAI()

encoding = tiktoken.get_encoding('cl100k_base')

cwd = Path.cwd()
data_path = cwd / 'data'
stored_queries_path = data_path / 'queries.csv'
curriculum_table_path = data_path / 'curriculums.txt'
curriculum_path = data_path / 'currics'

max_tokens = 8000
stored_queries = pd.DataFrame(columns = ['Name', 'Description', 'Embedding'])

class MenuOption(Enum):
    YES = 'Yes'
    NO = 'No'
    BACK = 'Back'
    EXIT = 'Exit'
    REMOVE = 'Remove'
    HISTORY = 'History'
    SKIP = 'I don\'t want to use this'
    SCAN = 'Scan'
    SCAN_DELETE = 'Scan and Delete after'
    QUERY = 'Query'


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

def read_sheet(path, delete_after=False):
	if not path.exists():
		return None

	extension = path.suffix

	df = None

	if extension == '.csv':
		df = pd.read_csv(path)
	elif extension in {'.xlsx', '.ods'}:
		df = pd.read_excel(path, sheet_name=None)

	if delete_after and df is not None:
		path.unlink()

	return df

def print_list(header, items, indeces=None, under_header=None, truncate_under=False):
	if under_header is not None:
		longest_line = max(header.split('\n'), key=len)

		n = ceil(len(longest_line) / len(under_header))
		under = under_header * n
		if truncate_under:
			under = under[:len(header)]

		header += '\n' + under

	padding = len(str(len(items)))

	if indeces is None:
		iterator = enumerate(items)
	else:
		iterator = zip(indeces, items)

	print(header)
	for i, item in iterator:
		print(f'{i+1:>{padding}}. {item}')

def print_list_and_query_input(header, items, under_header=None, truncate_under=False):
	max_accepted_input = len(items)
	is_valid = lambda x: x >= 1 and x <= max_accepted_input

	padding = len(str(len(items)))

	user_input = -1
	while not is_valid(user_input):
		clear()
		print_list(header, items, under_header=under_header, truncate_under=truncate_under)

		user_input = intify(input('Choose an option: '))

	return user_input

def scan_new_curriculums(path, delete_after=False):
	has_right_columns = lambda df: 'Description' in df.columns and 'Standard' in df.columns
	sanitize = lambda df: df[['Standard', 'Description']].dropna()

	files = filter(lambda p: p.is_file(), path.iterdir())

	dfs = []
	for child in files:
		maybe_df = read_sheet(child, delete_after=delete_after)

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
	text = string.replace('\n', ' ')
	return client.embeddings.create(input=text, model=model)
	

def establish_new_curriculums(named_dfs, already_used_names=set()):
	token_len = lambda x: len(encoding.encode(x))
	too_many_tokens = lambda df: df.Description.apply(token_len).gt(max_tokens).any()
	filtered_dfs = filter(lambda p: not too_many_tokens(p[1]), named_dfs)

	options = [MenuOption.YES.value, MenuOption.NO.value, MenuOption.SKIP.value]

	curriculum_table = open(curriculum_table_path, 'a')

	names_used = {name: 1 for name in already_used_names}

	for name, df in filtered_dfs:
		header = (
			f'New curriculum "{name}"\n'
			'Would you like to give it a different name?'
		)	

		selection = print_list_and_query_input(header, options, under_header=UNDER_ALL_HEADERS)
	
		selected_option = MenuOption(options[selection - 1])
		if selected_option == MenuOption.SKIP:
			continue
		elif selected_option == MenuOption.YES:
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
		df['embedding'] = df.Description.apply(lambda x: [len(x), -len(x)])

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
	print_list(header, curriculums, under_header=UNDER_ALL_HEADERS)
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
	print_list(header, [curriculums[i] for i in indeces], indeces=indeces, under_header=UNDER_ALL_HEADERS)

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

# TODO: list previously used PKs
def query_curriculum(curriculum):
	global stored_queries

	clear()

	table_path = curriculum_path / curriculum / 'table.csv'
	if not table_path.exists():
		print("ERROR: Curriculum file couldn't be found.")
		input("Press Enter to return to main menu...")
		return True

	df = pd.read_csv(table_path, index_col=[0])

	df['embedding'] = df.embedding.apply(literal_eval).apply(np.array)

	header = 'Would you like to save this query?'
	options = [MenuOption.YES.value, MenuOption.NO.value]
	selection = options[print_list_and_query_input(header, options) - 1]

	name = None
	if selection == MenuOption.YES.value:
		name = input("Enter name: ")
		while name in stored_queries["Name"].values:
			print("That name already exists. Try again.")
			name = input("Enter name: ")

	query = input("Enter query: ")

	query_embedding = [len(query), -len(query)]
	# TODO: When using real embeddings, remove above line and uncomment line below
	# query_embedding = embed_string(query)

	if name is not None:
		df2_dict = {'Name': [name], 'Description': [query], 'Embedding': [query_embedding]}
		df2 = pd.DataFrame.from_dict(df2_dict)
		stored_queries = pd.concat([stored_queries, df2], ignore_index=True)
		stored_queries.reset_index()

	query_embedding = np.array(query_embedding)

	df['similarity'] = df.embedding.apply(lambda x: abs(query_embedding[0] - x[0]))
	# TODO: When using real embeddings, remove above line and uncomment line below
	# df['similarity'] = df.embedding.apply(lambda x: distance.cosine(x, query_embedding))

	results = df.sort_values('similarity')[["Standard", "Description"]]

	matched_row = None

	for _, row in results.iterrows():
		header = row["Standard"]
		header += '\n' + '=' * len(header)

		clear()
		print(header)
		print(row["Description"])

		confirmation = input("Is this a good match? (Y)es or (N)o? ")

		if confirmation.lower()[0] == 'y':
			matched_row = row
			break

	if matched_row is None:
		print("It seems like there are no standards in this curriculum that match what you're looking for")
		input("Press Enter to return to the main menu.\n")
	else:
		# TODO: Store and write the match
		pass	

	return True

def curriculum_menu(curriculum):
	options = [MenuOption.QUERY.value, MenuOption.HISTORY.value, MenuOption.REMOVE.value, MenuOption.BACK.value]

	selected_number = print_list_and_query_input(curriculum, options, under_header=UNDER_ALL_HEADERS)
	selection = MenuOption(options[selected_number - 1])

	if selection == MenuOption.BACK:
		return True
	elif selection == MenuOption.REMOVE:
		confirmation = input('Are you sure? (Y)es or (N)o: ')
		if confirmation[0].lower() != 'y':
			return False

		shutil.rmtree(curriculum_path / curriculum)

		with open(curriculum_table_path, 'r') as f:
			lines = filter(lambda x: x != curriculum, map(lambda x: x[:-1], f.readlines()))

		with open(curriculum_table_path, 'w') as f:
			for line in lines:
				f.write(line + '\n')

		return True
	elif selection == MenuOption.QUERY:
		return query_curriculum(curriculum)
	else:
		# TODO: Implement history
		return True
		
def main_loop():
	items = [MenuOption.SCAN.value, MenuOption.SCAN_DELETE.value, MenuOption.REMOVE.value, MenuOption.EXIT.value]

	header = 'Please make a selection'

	current_selection = ''
	while current_selection != MenuOption.EXIT:
		curriculums = scan_for_curriculums()
		
		current_items = curriculums + items

		selection_number = print_list_and_query_input(header, current_items, under_header=UNDER_ALL_HEADERS)
		current_selection = current_items[selection_number - 1]
		try:
			current_selection = MenuOption(current_selection)

			if current_selection in {MenuOption.SCAN, MenuOption.SCAN_DELETE}:
				delete_after = current_selection == MenuOption.SCAN_DELETE
				currics = scan_new_curriculums(cwd, delete_after=delete_after)
				establish_new_curriculums(currics, set(curriculums))
			elif current_selection == MenuOption.REMOVE:
				status = False
				while not status:
					status = removing_menu(curriculums)
		except ValueError:
			status = False
			while not status:
				status = curriculum_menu(current_selection)

def map_queries_embedding(queries, f):
	return {name: {'Description': queries[name]['Description'], 'Embedding': f(queries[name]['Embedding'])} for name in queries}

if __name__ == '__main__':
	if not data_path.exists() or not data_path.is_dir():
		data_path.mkdir()
	if not stored_queries_path.exists():
		stored_queries_path.touch()
	if not curriculum_path.exists() or not curriculum_path.is_dir():
		curriculum_path.mkdir()

	try:
		stored_queries = pd.read_csv(stored_queries_path, index_col=[0])
		stored_queries['Embedding'] = stored_queries.Embedding.apply(np.array)
	except pd.errors.EmptyDataError:
		pass

	main_loop()

	stored_queries.to_csv(stored_queries_path)

