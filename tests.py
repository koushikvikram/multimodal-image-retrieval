'''Unit Tests for the Application'''

from src.dataset import Caption

CAPTION_PATH = "datasets/testing/test.txt"
c = Caption(CAPTION_PATH)

read_output = [
    'Go', '#follow', 'my', 
    '#official', 'artwork', 'page', 
    '(', '@rap_art', ')', 'finally', 
    'finished', '#Artwork', '#Alert', 
    '#workinprogress', '#50cent', 
    '#RapArt', 'by', '#ShonWil', 
    'on', '#Bristol', '#drawn', 
    'with', '#prismacolor', 
    '#prisma', '#marker', '#colorpencil', 
    '#newyork', '#detroit', 
    '#getrichordietryin', '#diamond', 
    '#art', '#fresh', '#fly', '#dope', 
    '#repost', '#tag', '#50cent', 
    '#artist', '#music', '#hiphop', 
    '#effen', '#effenvodka', '#power', 
    'tag', '@50cent', 'Thanks!']
clean_output = []
ID_OUTPUT = "test"
