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
stored_queries = pd.DataFrame(columns = ['Description', 'Embedding'], index=pd.Index([], name='Name'))

class MenuOption(Enum):
	YES = 'Yes'
	NO = 'No'
	BACK = 'Back'
	EXIT = 'Exit'
	EDIT = 'Edit'
	REMOVE = 'Remove'
	BEST = 'Mark as best so far and continue'
	HISTORY = 'History'
	SAVED = 'Saved'
	SKIP = 'I don\'t want to use this'
	SCAN = 'Scan'
	SCAN_DELETE = 'Scan and Delete after'
	QUERY = 'Query'
	REARRANGE = 'Rearrange'
	TEXT = 'Text'
	MAPPING = 'Mapping'

	@classmethod
	def from_value(cls, value):
		try:
			return cls(value)
		except ValueError:
			return None

illegal_names = {opt.value for opt in MenuOption}

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

def status_loop(call):
	status = False
	while not status:
		status = call()

def read_sheet(path, delete_after=False):
	if not path.exists():
		return None

	extension = path.suffix

	df = None

	if extension == '.csv':
		df = pd.read_csv(path, index_col="Standard")
	elif extension in {'.xlsx', '.ods'}:
		df = pd.read_excel(path, sheet_name=None, index_col="Standard")

	if delete_after and df is not None:
		path.unlink()

	return df

def get_curriculum_table_path(curriculum):
	return curriculum_path / curriculum / 'table.csv'

def get_curriculum_table(curriculum):
	path = get_curriculum_table_path(curriculum)

	table = pd.read_csv(path, index_col='Standard')
	table['embedding'] = table.embedding.apply(literal_eval).apply(np.array)

	return table

def empty_index(name):
	return pd.Index([], name=name)

def get_mapping_path(curriculum):
	return curriculum_path / curriculum / 'mappings.csv'

def get_mapping(curriculum):
	path = get_mapping_path(curriculum)
	print(path)
	if not path.exists():
		path.touch()
		return pd.DataFrame(columns=['Description', 'Standard', 'Standard Description'], index=empty_index('Name'))
	else:
		try:
			return pd.read_csv(path, index_col='Name')
		except pd.errors.EmptyDataError:
			return pd.DataFrame(columns=['Description', 'Standard', 'Standard Description'], index=empty_index('Name'))
		

def add_under_header(header, under_header, truncate=False):
    longest_line = max(header.split('\n'), key=len)

    n = ceil(len(longest_line) / len(under_header))
    under = under_header * n
    if truncate:
        under = under[:len(longest_line)]

    under = under[:os.get_terminal_size().columns]

    return header + '\n' + under

def print_list(header, items, indeces=None, under_header=None, truncate_under=False):
	if under_header is not None:
		header = add_under_header(header, under_header=under_header, truncate=truncate_under)

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
	has_right_columns = lambda df: 'Description' in df.columns and 'Standard' == df.index.name
	sanitize = lambda df: df[['Description']].dropna()

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
	return client.embeddings.create(input=text, model=model).data[0].embedding
	

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
		
		while name in illegal_names:
			print('This name is illegal, try again.')
			name = input('New name: ')

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

		# Get embeddings
		df['embedding'] = df.Description.apply(embed_string)

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

def query_ranking(df, query_embedding, include_similarity=False):
	df = df.copy()
	df['similarity'] = df.embedding.apply(lambda x: distance.cosine(x, query_embedding))
	df.sort_values('similarity', inplace=True)
	if include_similarity:
		return df[["Description", "similarity"]]
	else:
		return df[["Description"]]

def present_ranking(df):
	matched_row = None
	best_so_far = None

	for standard, row in df.iterrows():
		standard_header = f'{standard} ({row["similarity"]})'
		standard_header = add_under_header(standard_header, UNDER_ALL_HEADERS)

		desc_header = row["Description"]
		desc_header = add_under_header(desc_header, UNDER_ALL_HEADERS)

		header = f'{standard_header}\n{desc_header}\nIs this a good match?'

		options = [MenuOption.YES.value, MenuOption.NO.value, MenuOption.BEST.value, MenuOption.EXIT.value]

		selected_number = print_list_and_query_input(header, options)

		selection = MenuOption(options[selected_number - 1])
		if selection == MenuOption.YES:
			matched_row = row
			break
		elif selection == MenuOption.EXIT:
			break
		elif selection == MenuOption.BEST:
			best_so_far = row

	if matched_row is not None:
		return matched_row
	else:
		return best_so_far

def save_mapping(curriculum, mappings, name, desc, std, std_desc):
	entry_dict = {'Description': [desc], 'Standard': [std], 'Standard Description': [std_desc]}

	df = pd.DataFrame(entry_dict, index=pd.Index([name], name='Name'))

	mappings = pd.concat([mappings, df])
	mappings.to_csv(get_mapping_path(curriculum))

def query_curriculum(curriculum):
	global stored_queries

	clear()

	df = get_curriculum_table(curriculum)

	header = 'Would you like to save this query?'
	options = [MenuOption.YES.value, MenuOption.NO.value]
	selection = options[print_list_and_query_input(header, options) - 1]

	name = None
	if selection == MenuOption.YES.value:
		name = input("Enter name: ")
		while name in illegal_names or name in stored_queries.index:
			if name in illegal_names:
				print('That name is illegal. Try again.')
			else:
				print("That name already exists. Try again.")
			name = input("Enter name: ")

	query = input("Enter query: ")

	query_embedding = embed_string(query)

	if name is not None:
		df2_dict = {'Description': [query], 'Embedding': [query_embedding]}
		df2 = pd.DataFrame(df2_dict, index=pd.Index([name], name='Name'))
		stored_queries = pd.concat([stored_queries, df2])

	query_embedding = np.array(query_embedding)

	results = query_ranking(df, query_embedding, include_similarity=True)

	matched_row = present_ranking(results)

	if matched_row is None:
		print("It seems like there are no standards in this curriculum that match what you're looking for")
		input("Press Enter to return to the main menu.\n")
	else:
		mappings = get_mapping(curriculum)

		save_mapping(curriculum, mappings, name, query, matched_row.name, matched_row["Description"])

	return True

def history_entry(row):
	options = [MenuOption.BACK.value]

	name_header = row.name
	name_header = add_under_header(name_header, UNDER_ALL_HEADERS)

	standard_header = row['Standard']
	standard_header = add_under_header(standard_header, UNDER_ALL_HEADERS)

	header = f'{name_header}\n{row["Description"]}\n\n{standard_header}\n{row["Standard Description"]}'
	selection_number = print_list_and_query_input(header, options, under_header=UNDER_ALL_HEADERS)

	if MenuOption(options[selection_number - 1]) == MenuOption.BACK:
		return True

def curriculum_history(curriculum_name, mappings):
	options = [MenuOption.BACK.value]
	
	names = mappings.index.tolist()
	standards = mappings.Standard.tolist()
	prepend = [f'{name} -> {std}' for name, std in zip(names, standards)]

	options = prepend + options
    
	header = f"Items previously matched to {curriculum_name}"

	selection = None
	while selection != MenuOption.BACK:
		selected_number = print_list_and_query_input(header, options, under_header=UNDER_ALL_HEADERS)
		selection = MenuOption.from_value(options[selected_number - 1])

		if selection is None:
			call_history_entry = lambda: history_entry(mappings.iloc[selected_number - 1])
			status_loop(call_history_entry)

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
		mappings = get_mapping(curriculum)

		return curriculum_history(curriculum, mappings)

def swap_menu():
	curriculums = scan_for_curriculums()
	header = (
			"Type in the numbers of two curriculums you'd like to swap.\n"
			"Leave blank and press Enter to go back."
	)

	clear()
	print_list(header, curriculums, under_header=UNDER_ALL_HEADERS)
	choice = input('Enter here: ')

	if len(choice) == 0:
		return True

	choices = choice.split()[:2]
	if len(choices) < 2:
		return False

	try:
		a, b = tuple(map(int, choices))
		curriculums[a - 1], curriculums[b - 1] = curriculums[b - 1], curriculums[a - 1]
		with open(curriculum_table_path, 'w') as f:
			for line in curriculums:
				f.write(line + '\n')
		return False
	except ValueError:
		return False

def saved_query_new_query_menu(entry):
	curriculums = scan_for_curriculums()

	options = [MenuOption.BACK.value]

	options = curriculums + options

	header = f'Choose curriculum for {entry.name}'
	selection_number = print_list_and_query_input(header, options, under_header=UNDER_ALL_HEADERS)
	
	selection = MenuOption.from_value(options[selection_number - 1])
	if selection == MenuOption.BACK:
		return True
	else:
		chosen_curriculum = curriculums[selection_number - 1]

		mappings = get_mapping(chosen_curriculum)

		if entry.name in mappings.index:
			mapping_entry = mappings.loc[entry.name]

			header = f'There is already a mapping for {entry.name} in {chosen_curriculum}.\n\n'
			
			header += add_under_header(mapping_entry["Standard"], UNDER_ALL_HEADERS)

			standard_desc = mapping_entry["Standard Description"]
			standard_desc = add_under_header(standard_desc, UNDER_ALL_HEADERS)

			header = f'{header}\n{standard_desc}\nWould you like to change it?'

			options = [MenuOption.YES.value, MenuOption.NO.value]

			selection_number = print_list_and_query_input(header, options)

			if MenuOption(options[selection_number - 1]) == MenuOption.NO:
				return False

			mappings.drop(entry.name, inplace=True)

		curriculum_table = get_curriculum_table(chosen_curriculum)

		embedding = np.array(entry["Embedding"])
		ranking = query_ranking(curriculum_table, embedding, include_similarity=True)

		matching_row = present_ranking(ranking)

		if matching_row is not None:
			save_mapping(chosen_curriculum, mappings, entry.name, entry["Description"], matching_row.name, matching_row["Description"])
			return True

		return True


def saved_query_menu(entry):
	# TODO: Option to see previous mappings
	options = [MenuOption.QUERY.value, MenuOption.BACK.value]

	selection_number = print_list_and_query_input(entry.name, options, under_header=UNDER_ALL_HEADERS)

	selection = MenuOption(options[selection_number - 1])

	if selection == MenuOption.BACK:
		return True
	else:
		call = lambda: saved_query_new_query_menu(entry)
		status_loop(call)
		return False

def saved_menu():
	options = [MenuOption.BACK.value]

	options = stored_queries.index.tolist() + options

	selection_number = print_list_and_query_input('Saved', options, under_header=UNDER_ALL_HEADERS)
	selection = MenuOption.from_value(options[selection_number - 1])

	if selection == MenuOption.BACK:
		return True
	else:
		row = stored_queries.iloc[selection_number - 1]
		status_loop(lambda: saved_query_menu(row))
		return False

def main_loop():
	items = [MenuOption.SCAN.value, MenuOption.SCAN_DELETE.value, MenuOption.SAVED.value, MenuOption.REARRANGE.value, MenuOption.REMOVE.value, MenuOption.EXIT.value]

	header = 'Please make a selection'

	current_selection = None
	while current_selection != MenuOption.EXIT:
		curriculums = scan_for_curriculums()
		
		current_items = curriculums + items

		selection_number = print_list_and_query_input(header, current_items, under_header=UNDER_ALL_HEADERS)
		current_selection = MenuOption.from_value(current_items[selection_number - 1])

		if current_selection in {MenuOption.SCAN, MenuOption.SCAN_DELETE}:
			delete_after = current_selection == MenuOption.SCAN_DELETE
			currics = scan_new_curriculums(cwd, delete_after=delete_after)
			establish_new_curriculums(currics, set(curriculums))
		elif current_selection == MenuOption.REMOVE:
			status_loop(lambda: removing_menu(curriculums))
		elif current_selection == MenuOption.REARRANGE:
			status_loop(swap_menu)
		elif current_selection == MenuOption.SAVED:
			status_loop(saved_menu)
		elif current_selection is None:
			current_selection = current_items[selection_number - 1]
			status_loop(lambda: curriculum_menu(current_selection))

if __name__ == '__main__':
	if not data_path.exists() or not data_path.is_dir():
		data_path.mkdir()
	if not stored_queries_path.exists():
		stored_queries_path.touch()
	if not curriculum_path.exists() or not curriculum_path.is_dir():
		curriculum_path.mkdir()

	try:
		stored_queries = pd.read_csv(stored_queries_path, index_col='Name')
		stored_queries["Embedding"] = stored_queries.Embedding.apply(literal_eval)
	except pd.errors.EmptyDataError:
		pass

	main_loop()

	stored_queries.to_csv(stored_queries_path)

