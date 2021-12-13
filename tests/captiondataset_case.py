'''Test case variables for CaptionDataset'''

import numpy as np

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

CLEAN_EMBEDDINGS_RESULT = {
    '000001': np.array([0.48158005, 0.56169397, 0.6302556 , 0.50096285, 0.6406477 ,
        0.5665123 , 0.6391689 , 0.5362362 , 0.8927736 , 0.72586244,
        0.2851158 , 0.7206787 , 0.54273397, 0.7055956 , 0.8629575 ,
        0.6900076 , 0.6813975 , 0.66263366, 0.80908406, 0.4651312 ,
        1.        , 0.47737682, 0.460142  , 0.77637154, 0.5030053 ,
        0.42341596, 0.52811825, 0.6800097 , 0.33794582, 0.07935591,
        0.9076179 , 0.41943896, 0.438199  , 0.90291786, 0.8264345 ,
        0.8373191 , 0.40113991, 0.70827127, 0.41452292, 0.40696168,
        0.36159223, 0.44060504, 0.83998406, 0.5498376 , 0.66401094,
        0.50747013, 0.46865425, 0.67850274, 0.561077  , 0.50417405,
        0.31372207, 0.41896302, 0.7745291 , 0.63663256, 0.6299229 ,
        0.36152178, 0.80526   , 0.6142762 , 0.680691  , 0.7267224 ,
        0.5215627 , 0.7500492 , 0.6660475 , 0.7452486 , 0.81529224,
        0.9101264 , 0.6116001 , 0.52220076, 0.53820664, 0.46650568,
        0.5185653 , 0.6545208 , 0.44063047, 0.7303986 , 0.6789377 ,
        0.6743403 , 0.        , 0.547153  , 0.4436783 , 0.95102936,
        0.68505114, 0.45342737, 0.5588117 , 0.61354625, 0.7472519 ,
        0.37755966, 0.45979095, 0.49229234, 0.51288897, 0.70332503,
        0.57959926, 0.546241  , 0.6023976 , 0.7460788 , 0.55945885,
        0.5799856 , 0.7131122 , 0.39636457, 0.5698697 , 0.84205955],
       dtype=np.float32),
    '000002': np.array([0.39461973, 0.6906092 , 0.5815087 , 0.7252795 , 0.6622203 ,
        0.47909963, 0.53481686, 0.46807134, 0.8216209 , 0.5773744 ,
        0.18957114, 0.6648622 , 0.5624246 , 0.56660134, 0.8489548 ,
        0.57658046, 0.6107128 , 0.5951283 , 0.7337138 , 0.28522754,
        1.        , 0.45666647, 0.37071165, 0.647348  , 0.54104835,
        0.42777917, 0.43883464, 0.6844845 , 0.2964369 , 0.08488819,
        0.9382527 , 0.40225574, 0.40082052, 0.74608016, 0.7655614 ,
        0.74306315, 0.47480088, 0.717049  , 0.24457325, 0.28610253,
        0.383717  , 0.25012287, 0.7896519 , 0.48508075, 0.4717878 ,
        0.57110125, 0.44060126, 0.7707842 , 0.4867341 , 0.35954073,
        0.22880337, 0.44546255, 0.6690933 , 0.57510287, 0.58944106,
        0.2973398 , 0.8332815 , 0.6409015 , 0.7101045 , 0.63829786,
        0.48961213, 0.5945013 , 0.6444266 , 0.727155  , 0.7886844 ,
        0.87311506, 0.5387276 , 0.47468394, 0.5648535 , 0.34138992,
        0.4701219 , 0.5960677 , 0.3878743 , 0.683331  , 0.52422196,
        0.6172886 , 0.        , 0.451021  , 0.3249259 , 0.9575583 ,
        0.6168104 , 0.41743058, 0.45854914, 0.64570177, 0.6587471 ,
        0.33942896, 0.34156153, 0.47599795, 0.46434164, 0.74339503,
        0.53972274, 0.409504  , 0.63711476, 0.6545799 , 0.48925507,
        0.6599883 , 0.5874424 , 0.26175818, 0.5103996 , 0.7711546 ],
       dtype=np.float32),
    '000003': np.array([0.3832451 , 0.28136823, 0.53973866, 0.3510817 , 0.8268399 ,
        0.3872593 , 0.44179204, 0.18799278, 0.07196975, 0.        ,
        0.3704926 , 0.15450941, 0.3943315 , 0.617742  , 0.13021453,
        0.435348  , 0.7105198 , 0.6604243 , 0.4079153 , 0.31067994,
        0.31041807, 0.038235918, 0.4805433 , 0.25130975, 0.50592816,
        0.5227454 , 0.2280244 , 0.6007142 , 0.47399938, 0.53346086,
        0.46812007, 0.29676878, 0.63075036, 0.51054716, 0.31931213,
        0.25577813, 0.39113525, 0.56980217, 0.51255447, 0.21303713,
        0.25195196, 0.42816702, 0.23693722, 0.39355952, 0.3292831 ,
        0.588118  , 0.3284439 , 0.08168511, 0.6152355 , 0.38739038,
        0.4053094 , 0.21178375, 0.3767129 , 0.2883389 , 0.3231845 ,
        0.30124706, 0.31681994, 0.58677787, 0.5811582 , 0.4468231 ,
        0.61460185, 1.        , 0.27524576, 0.6711838 , 0.38243896,
        0.71588683, 0.13369738, 0.2799586 , 0.29198858, 0.21250513,
        0.34058967, 0.20407237, 0.42072153, 0.5193067 , 0.092829585,
        0.47014636, 0.45260912, 0.21514896, 0.022359211, 0.49212605,
        0.439529  , 0.48085058, 0.3674651 , 0.22281764, 0.13216105,
        0.47575334, 0.3407896 , 0.30130804, 0.26096234, 0.33337256,
        0.5237136 , 0.523834  , 0.323323  , 0.46194842, 0.10299653,
        0.16668725, 0.3400978 , 0.35747498, 0.5571357 , 0.45070672],
       dtype=np.float32)
       }

WORD2VEC_DATASET_RESULT = [
    [
        'coffee',
        'coffee',
    ],
    [
        'toronto',
        'toronto',
        'montreal',
    ],
    [
        'montreal',
    ],
]

CLEAN_SPLIT_NO_SHUFFLE = (
    {
        '000001': [
            'toronto',
            'toronto',
            'montreal',
            ]
    },
    {
        '000002': [
            'montreal',
            ]
    },
    {
        '000003': [
            'coffee',
            'coffee',
            ]
    }
)