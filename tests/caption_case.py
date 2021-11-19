'''Test cases for test_captions.py'''

FILE_NAMES = [
    'caps.txt',
    'empty.txt',
    'hindi.txt',
    'mixed_language.txt',
    'newline.txt',
    'no_stopwords.txt',
    'only_spaces.txt',
    'only_stopwords.txt',
    'single_word.txt',
    'spaced_letters.txt',
    'special_characters.txt',
    'special_stop.txt',
    'stop_no_stop.txt',
    'with_special_characters.txt',
    ]

INCORRECT_EXT_FILES = [
    'no_file_extension',
    'non_txt.json',
]

NON_EXISTING_FILES = [
    'no_such_file.txt',
]

RAW_CAPTIONS = [
    ['DONT', 'HOLD', 'SHIFT', 'KEY'],
    [''],
    ['हिंदी', 'में', 'लिखो'],
    ['हिंदी', 'English'],
    ['line1\n\nline2'],
    ['coffee', 'tree', 'guy', 'travel', 'rain', 'hot', 'paint', 'toronto'],
    ['', '', '', '', ''],
    ['http', 'https', 'photo', 'but', 'than', 'picture', 'image', 'insta', 'instagram', 'post'],
    ['word'],
    ['a', 'b', 'c'],
    ['#@!^&*$'],
    ['#than', '@photo'],
    ['http', 'coffee', 'than', 'post', 'tree'],
    ['#Artwork', '#Alert', '#workinprogress', 'tag', '@50cent', 'Thanks!'],
]

CLEAN_CAPTIONS = [
    ['dont', 'hold', 'shift', 'key'],
    [],
    [],
    ['english'],
    ['line1', 'line2'],
    ['coffee', 'tree', 'guy', 'travel', 'rain', 'hot', 'paint', 'toronto'],
    [],
    [],
    ['word'],
    ['b', 'c'],
    [],
    [],
    ['coffee', 'tree'],
    ['artwork', 'alert', 'workinprogress', 'tag', '50cent', 'thanks'],
]

FILE_ID = [
    'caps',
    'empty',
    'hindi',
    'mixed_language',
    'newline',
    'no_stopwords',
    'only_spaces',
    'only_stopwords',
    'single_word',
    'spaced_letters',
    'special_characters',
    'special_stop',
    'stop_no_stop',
    'with_special_characters',
]
