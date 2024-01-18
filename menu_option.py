from enum import Enum

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