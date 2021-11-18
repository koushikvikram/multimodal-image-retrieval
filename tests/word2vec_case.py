'''Test Cases'''

VOCAB_WORDS = [
    "newyork",
    "nyc",
    "club",
    "night",
    "outdoor",
    "travel",
    "fun",
    "delicious"
]

STOP_WORDS = [
    'http',
    'https',
    'photo',
    'picture',
    'image',
    'insta',
    'instagram',
    'post'
]

NON_MATCHING_PAIRS = [
    ("breakfast cereal lunch dinner".split(), "cereal"),
    ("spoon saturday friday sunday".split(), "spoon"),
    ("toronto montreal vancouver newyork".split(), "newyork"),
    ("paint art boy".split(), "boy"),
    ("hot cool sexy fun".split(), "fun"),
    ("rain snow wind street".split(), "street"),
    ("travel adventure fun work".split(), "work"),
    ("guy man dude girl".split(), "girl"),
    ("park tree forest museum".split(), "museum"),
    ("coffee latte espresso burger".split(), "burger"),
]

SIMILAR_WORDS = [
    ("breakfast", "lunch"),
    ("friday", "sunday"),
    ("toronto", "montreal"),
    ("paint", "art"),
    ("hot", "sexy"),
    ("rain", "snow"),
    ("travel", "adventure"),
    ("guy", "man"),
    ("tree", "forest"),
    ("coffee", "latte"),
]
