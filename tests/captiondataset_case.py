'''Test case variables for CaptionDataset'''

UNCLEAN_READ_RESULT = {
    '000001': [
        'http:',
        'insta',
        '#post',
        'toronto',
        'but',
        'https:',
        'insta',
        'image',
        'toronto',
        'montreal',
        ],
    '000002': [
        'photo',
        'insta',
        'than',
        '#post',
        'montreal',
        'instagram',
        ],
    '000003': [
        'coffee',
        'coffee',
        ]
    }

CLEAN_READ_RESULT = {
    '000001': [
        'toronto',
        'toronto',
        'montreal',
        ],
    '000002': [
        'montreal',
        ],
    '000003': [
        'coffee',
        'coffee',
        ]
    }

UNCLEAN_MIN_COUNT_3_RESULT = {
    '000001': [
               'insta',
               'insta',
               ],
    '000002': [
               'insta',
               ]
     }

CLEAN_MIN_COUNT_2_RESULT = {
    '000001': [
               'toronto',
               'toronto',
               'montreal',
               ],
    '000002': [
               'montreal',
               ],
    '000003': [
               'coffee',
               'coffee',
               ]
     }