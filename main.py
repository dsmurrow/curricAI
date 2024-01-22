"""Main module, most functions here are to do with displaying queries for the user."""

from ast import literal_eval
from collections.abc import Iterable
import os
from pathlib import Path
import re
import shutil

from math import ceil
import multiprocessing as mp
import numpy as np
import pandas as pd
import tiktoken

from ai_calls import embed_string
from baked_paths import CURRICULUM_DIR_PATH, CURRICULUM_LIST_PATH, CWD, \
    DATA_PATH, STORED_QUERIES_PATH
from curriculum import Curriculum
from menu_option import MenuOption

MAX_TOKENS = 8000
UNDER_ALL_HEADERS = '=' * 3

encoding = tiktoken.get_encoding('cl100k_base')

stored_queries = \
    pd.DataFrame(columns=['Description', 'Embedding'], index=pd.Index([], name='Name'))

illegal_names = {opt.value for opt in MenuOption}

curriculum_map = {}


def clear():
    """OS-agnostic function to clear terminal"""
    name = os.name

    if name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def intify(x) -> int:
    """Return x as int if possible, else return -1 """
    try:
        return int(x)
    except ValueError:
        return -1


def status_loop(call):
    """Loop function until it returns a truthy value"""
    status = False
    while not status:
        status = call()

    return status


def read_sheet(path: Path, delete_after: bool = False):
    """Read curriculum-formatted datasheet"""
    if not path.exists():
        return None

    extension = path.suffix

    df = None

    if extension == '.csv':
        df = pd.read_csv(path, index_col="Standard")
    elif extension in {'.xlsx', '.ods'}:
        # Reads all excel sheets, must all be compatible or program will fail
        df = pd.read_excel(path, sheet_name=None, index_col="Standard")

    if delete_after and df is not None:
        path.unlink()

    return df


def get_curriculum_object(name: str):
    """Lazily-initialized curriculum objects"""
    if name not in curriculum_map:
        curriculum_map[name] = Curriculum(name)

    return curriculum_map[name]


def add_under_header(header: str, under_header: str, truncate: bool = False):
    """Format text to put under_header under the header"""
    longest_line = max(header.split('\n'), key=len)

    n = ceil(len(longest_line) / len(under_header))
    under = under_header * n
    if truncate:
        under = under[:len(longest_line)]

    under = under[:os.get_terminal_size().columns]

    return header + '\n' + under


def print_list(header: str, items: Iterable[str], *, indeces: Iterable[str] = None,
               under_header: str = None, truncate_under: bool = False):
    """Print list of items"""
    if under_header is not None:
        header = add_under_header(
            header, under_header=under_header, truncate=truncate_under)

    padding = len(str(len(items)))

    if indeces is None:
        iterator = enumerate(items)
    else:
        iterator = zip(indeces, items)

    print(header)
    for i, item in iterator:
        print(f'{i+1:>{padding}}. {item}')


def print_list_and_query_input(header: str, items: Iterable[str],
                               *, under_header: str = None, truncate_under: str = False):
    """Print list of items and then go into a strict input loop \
        that only accepts the numbers of the given options"""
    max_accepted_input = len(items)

    def is_valid(x):
        return x >= 1 and x <= max_accepted_input

    user_input = -1
    while not is_valid(user_input):
        clear()
        print_list(header, items, under_header=under_header,
                   truncate_under=truncate_under)

        user_input = intify(input('Choose an option: '))

    return user_input


def scan_new_curriculums(path: Path, delete_after=False):
    """Scan and parse spreadsheets in cwd"""
    def has_right_columns(df):
        return 'Description' in df.columns and 'Standard' == df.index.name

    def sanitize(df):
        return df[['Description']].dropna()

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


def establish_new_curriculums(named_dfs: list[pd.DataFrame], already_used_names=None):
    """Sanitize scanned DataFrames and settle them into the system"""
    if already_used_names is None:
        already_used_names = set()

    def token_len(x):
        return len(encoding.encode(x))

    def too_many_tokens(df):
        return df.Description.apply(token_len).gt(MAX_TOKENS).any()
    filtered_dfs = filter(lambda p: not too_many_tokens(p[1]), named_dfs)

    options = [MenuOption.YES.value,
               MenuOption.NO.value, MenuOption.SKIP.value]

    curriculum_table = open(CURRICULUM_LIST_PATH, 'a', encoding='utf-8')

    names_used = {name: 1 for name in already_used_names}

    pool = mp.Pool(mp.cpu_count())
    result_list = []

    for name, df in filtered_dfs:
        header = (
            f'New curriculum "{name}"\n'
            'Would you like to give it a different name?'
        )

        selection = print_list_and_query_input(
            header, options, under_header=UNDER_ALL_HEADERS)

        selected_option = MenuOption(options[selection - 1])
        if selected_option == MenuOption.SKIP:
            continue

        if selected_option == MenuOption.YES:
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

        # Make directories for curriculums
        curriculum_dir = CURRICULUM_DIR_PATH / name
        curriculum_table.write(name + '\n')

        curriculum_dir.mkdir()

        # Get embeddings
        result_list.append(pool.apply_async(Curriculum, (name, df,)))
    pool.close()

    for result in result_list:
        result.get().save()

    curriculum_table.close()


def scan_for_curriculums():
    """Get curriculum list."""
    if not CURRICULUM_LIST_PATH.exists() or not CURRICULUM_LIST_PATH.is_file():
        CURRICULUM_LIST_PATH.touch()

    with open(CURRICULUM_LIST_PATH, 'r', encoding='utf-8') as file:
        currics = list(map(lambda x: x[:-1], file))

    return currics


def removal_query(header: str, items: Iterable[str], under_header: str):
    """Generic menu for removing any set of items."""
    valid_indeces = range(1, len(items) + 1)

    while True:
        clear()
        print_list(header, items, under_header=under_header)
        selections = input('Selections: ').split()

        if len(selections) == 0:
            return []
        elif len(selections) > 0 and selections[0] == '*':
            selections = list(valid_indeces)

        index_set = set()
        for item in selections:
            try:
                index_set.add(int(item))
            except ValueError:
                input('Invalid input. Press Enter to try again\n')
                continue

        indeces = filter(lambda x: x in valid_indeces, index_set)
        indeces = list(map(lambda x: x - 1, indeces))
        return indeces


def removing_menu(curriculums: list[str]):
    """When the user selects the 'Remove' option in the main menu."""
    header = (
        'Write the numbers corresponding to the curriculums you\'d like to remove.\n'
        'Separate entries by leaving space between them.\n'
        'Type \'*\' to remove all curriculums.\n'
        'Leave blank if you\'d like to leave this menu.'
    )

    indeces = removal_query(header, curriculums,
                            under_header=UNDER_ALL_HEADERS)

    if len(indeces) == 0:
        return True

    clear()
    header = 'Items to be deleted.'
    print_list(header, [curriculums[i] for i in indeces],
               indeces=indeces, under_header=UNDER_ALL_HEADERS)

    selection = input('Delete these items? (Y)es or (N)o: ')
    if selection[0].lower() != 'y':
        return False

    indeces.sort(reverse=True)

    for i in indeces:
        name = curriculums[i]

        curriculum_dir = CURRICULUM_DIR_PATH / name

        shutil.rmtree(curriculum_dir)

        del curriculums[i]

    with open(CURRICULUM_LIST_PATH, 'w', encoding='utf-8') as f:
        for curriculum in curriculums:
            f.write(curriculum + '\n')

    return True


def present_ranking(df: pd.DataFrame):
    """Present the curriculum standards of order of their similarity with a query."""
    matched_row = None
    best_so_far = None

    for standard, row in df.iterrows():
        standard_header = f'{standard} ({row["similarity"]})'
        standard_header = add_under_header(standard_header, UNDER_ALL_HEADERS)

        desc_header = row["Description"]
        desc_header = add_under_header(desc_header, UNDER_ALL_HEADERS)

        header = f'{standard_header}\n{desc_header}\nIs this a good match?'

        options = [MenuOption.YES.value, MenuOption.NO.value,
                   MenuOption.BEST.value, MenuOption.EXIT.value]

        selected_number = print_list_and_query_input(header, options)

        selection = MenuOption(options[selected_number - 1])
        if selection == MenuOption.YES:
            matched_row = row
            break
        elif selection == MenuOption.EXIT:
            break
        elif selection == MenuOption.BEST:
            best_so_far = row

    if matched_row is None:
        return best_so_far

    return matched_row


def query_curriculum(curriculum: Curriculum):
    """Give the curriculum a query to find a good match."""
    global stored_queries

    clear()

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

    results = curriculum.query(query_embedding, include_similarity=True)

    matched_row = present_ranking(results)

    if matched_row is None:
        print("It seems like there are no standards in this curriculum that match what you're looking for")
        input("Press Enter to return to the main menu.\n")
    else:
        curriculum.mapping.add_mapping(
            name, query, matched_row.name, matched_row['Description'])

    return True


def history_entry(row: pd.Series):
    """Menu to see the details of a chosen mapping."""
    options = [MenuOption.BACK.value]

    name_header = row.name
    name_header = add_under_header(name_header, UNDER_ALL_HEADERS)

    standard_header = row['Standard']
    standard_header = add_under_header(standard_header, UNDER_ALL_HEADERS)

    header = (
        f'{name_header}\n{row["Description"]}\n\n'
        f'{standard_header}\n{row["Standard Description"]}'
    )
    selection_number = print_list_and_query_input(
        header, options, under_header=UNDER_ALL_HEADERS)

    if MenuOption(options[selection_number - 1]) == MenuOption.BACK:
        return True

    return False


def curriculum_history(curriculum: Curriculum):
    """The menu for a chosen curriculum's query mappings."""
    baked_options = [MenuOption.REMOVE.value, MenuOption.BACK.value]

    selection = None
    while selection != MenuOption.BACK:
        names = curriculum.mapping.names
        standards = curriculum.mapping.standards

        prepend = [f'{name} -> {std}' for name, std in zip(names, standards)]

        options = prepend + baked_options

        header = f"Items previously matched to {curriculum.name}"

        selected_number = print_list_and_query_input(
            header, options, under_header=UNDER_ALL_HEADERS)
        selection = MenuOption.from_value(options[selected_number - 1])

        if selection is None:
            def call_history_entry():
                return history_entry(curriculum.mapping[selected_number - 1])
            status_loop(call_history_entry)
        elif selection is MenuOption.REMOVE:
            header = (
                'Type out the numbers corresponding to the mappings you\'d like to remove.\n'
                'Leave a space between each entry.\n'
                'Leave empty to go back. Type only \'*\' to delete all entries.'
            )

            indeces = removal_query(
                header, prepend, under_header=UNDER_ALL_HEADERS)

            if len(indeces) == 0:
                continue

            clear()
            print_list('Items to be deleted:',
                       [prepend[i] for i in indeces],
                       indeces=indeces,
                       under_header=UNDER_ALL_HEADERS)
            confirmation = input('Delete these items? (Y)es or (N)o?: ')

            if len(confirmation) > 0 and confirmation[0].lower() != 'y':
                continue

            for name in [names[i] for i in indeces]:
                curriculum.mapping.remove(name)

    return True


def curriculum_menu(curriculum: Curriculum):
    """Menu for when the user selects one of the saved curriculums."""
    options = [MenuOption.QUERY.value, MenuOption.HISTORY.value,
               MenuOption.REMOVE.value, MenuOption.BACK.value]

    selected_number = print_list_and_query_input(
        curriculum.name, options, under_header=UNDER_ALL_HEADERS)
    selection = MenuOption(options[selected_number - 1])

    if selection == MenuOption.BACK:
        return True
    if selection == MenuOption.REMOVE:
        confirmation = input('Are you sure? (Y)es or (N)o: ')
        if confirmation[0].lower() != 'y':
            return False

        shutil.rmtree(curriculum.directory)

        with open(CURRICULUM_LIST_PATH, 'r', encoding='utf-8') as f:
            lines = filter(lambda x: x != curriculum.name, map(
                lambda x: x[:-1], f.readlines()))

        with open(CURRICULUM_LIST_PATH, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

        return True
    if selection == MenuOption.QUERY:
        return query_curriculum(curriculum)

    def call():
        return curriculum_history(curriculum)

    status_loop(call)

    return False


def swap_menu():
    """The 'Swap' menu option."""
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
        curriculums[a - 1], curriculums[b - 1] = \
            curriculums[b - 1], curriculums[a - 1]
        with open(CURRICULUM_LIST_PATH, 'w', encoding='utf-8') as f:
            for line in curriculums:
                f.write(line + '\n')
        return False
    except ValueError:
        return False


def saved_query_new_query_menu(entry: pd.Series):
    """When the user, from the saved query menu, chooses to query a different curriculum."""
    curriculums = scan_for_curriculums()

    options = [MenuOption.BACK.value]

    options = curriculums + options

    header = f'Choose curriculum for {entry.name}'
    selection_number = print_list_and_query_input(
        header, options, under_header=UNDER_ALL_HEADERS)

    selection = MenuOption.from_value(options[selection_number - 1])
    if selection == MenuOption.BACK:
        return True

    curriculum = get_curriculum_object(curriculums[selection_number - 1])

    if entry.name in curriculum.mapping.names:
        mapping_entry = curriculum.mapping[entry.name]

        header = f'There is already a mapping for {entry.name} in {curriculum.name}.\n\n'

        header += add_under_header(
            mapping_entry["Standard"], UNDER_ALL_HEADERS)

        standard_desc = mapping_entry["Standard Description"]
        standard_desc = add_under_header(standard_desc, UNDER_ALL_HEADERS)

        header = f'{header}\n{standard_desc}\nWould you like to change it?'

        options = [MenuOption.YES.value, MenuOption.NO.value]

        selection_number = print_list_and_query_input(header, options)

        if MenuOption(options[selection_number - 1]) == MenuOption.NO:
            return False

        curriculum.mapping.remove(entry.name)

        embedding = np.array(entry["Embedding"])
        ranking = curriculum.query(embedding, include_similarity=True)

        matching_row = present_ranking(ranking)

        if matching_row is not None:
            curriculum.mapping.add_mapping(
                entry.name, entry['Description'], matching_row.name, matching_row['Description'])
            return True

    return False


def saved_query_menu(entry: pd.Series):
    """When the user selects a saved query."""

    # TODO: Option to see previous mappings
    options = [MenuOption.QUERY.value, MenuOption.BACK.value]

    selection_number = \
        print_list_and_query_input(entry.name, options, under_header=UNDER_ALL_HEADERS)

    selection = MenuOption(options[selection_number - 1])

    if selection == MenuOption.BACK:
        return True

    def call():
        return saved_query_new_query_menu(entry)
    status_loop(call)
    return False


def saved_menu():
    """What happens when user selects the 'Saved' option."""
    options = [MenuOption.BACK.value]

    options = stored_queries.index.tolist() + options

    selection_number = \
        print_list_and_query_input('Saved', options, under_header=UNDER_ALL_HEADERS)
    selection = MenuOption.from_value(options[selection_number - 1])

    if selection == MenuOption.BACK:
        return True

    row = stored_queries.iloc[selection_number - 1]
    status_loop(lambda: saved_query_menu(row))
    return False


def main_loop():
    """Main menu of the program."""

    items = [MenuOption.SCAN.value, MenuOption.SCAN_DELETE.value, MenuOption.SAVED.value,
             MenuOption.REARRANGE.value, MenuOption.REMOVE.value, MenuOption.EXIT.value]

    header = 'Please make a selection'

    current_selection = None
    while current_selection != MenuOption.EXIT:
        curriculums = scan_for_curriculums()

        current_items = curriculums + items

        selection_number = \
            print_list_and_query_input(header, current_items, under_header=UNDER_ALL_HEADERS)
        current_selection = \
            MenuOption.from_value(current_items[selection_number - 1])

        if current_selection in {MenuOption.SCAN, MenuOption.SCAN_DELETE}:
            delete_after = current_selection == MenuOption.SCAN_DELETE
            currics = scan_new_curriculums(CWD, delete_after=delete_after)
            establish_new_curriculums(currics, set(curriculums))
        elif current_selection == MenuOption.REMOVE:
            status_loop(lambda: removing_menu(curriculums))
        elif current_selection == MenuOption.REARRANGE:
            status_loop(swap_menu)
        elif current_selection == MenuOption.SAVED:
            status_loop(saved_menu)
        elif current_selection is None:
            current_selection = current_items[selection_number - 1]
            curriculum = get_curriculum_object(current_selection)
            status_loop(lambda: curriculum_menu(curriculum))


if __name__ == '__main__':
    if not DATA_PATH.exists() or not DATA_PATH.is_dir():
        DATA_PATH.mkdir()
    if not STORED_QUERIES_PATH.exists():
        STORED_QUERIES_PATH.touch()
    if not CURRICULUM_DIR_PATH.exists() or not CURRICULUM_DIR_PATH.is_dir():
        CURRICULUM_DIR_PATH.mkdir()

    try:
        stored_queries = pd.read_csv(STORED_QUERIES_PATH, index_col='Name')
        stored_queries["Embedding"] = \
            stored_queries.Embedding.apply(literal_eval)
    except pd.errors.EmptyDataError:
        pass

    main_loop()

    for obj in curriculum_map.values():
        obj.mapping.save()

    stored_queries.to_csv(STORED_QUERIES_PATH)
