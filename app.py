'''Web app for Multimodal Image Retrieval'''
import pickle
from PIL import Image
import numpy as np
from gensim.models import Word2Vec
import streamlit as st

import config.app as cfg


# open embeddings for images generated from the CNN
with open(cfg.EMBEDDINGS_FILE_PATH, "rb") as file:
    embeddings_dict = pickle.load(file)

# images directory
images_dir = cfg.IMAGES_DIR

# load our Word2Vec model
model = Word2Vec.load(cfg.WORD2VEC_MODEL)

def get_closest_image(positive, negative):
    # make single vector representation
    num_captions = len(positive) + len(negative)
    embedding = np.zeros(100)
    if num_captions == 0:
        raise Exception("No captions given")
    for positive_caption in positive:
        embedding += model.wv[positive_caption]
    if len(negative) == 1:
        embedding -= model.wv[negative[0]]
    # get average of embeddings
    embedding /= num_captions
    # normalize embeddings
    if min(embedding) < 0:
        embedding = embedding - min(embedding)
    if max(embedding) > 0:
        embedding = embedding / max(embedding)
    # compute distance between each image embedding and query embedding
    similarities = {}
    for image_name, image_embedding in embeddings_dict.items():
        similarities[image_name] = np.dot(image_embedding, embedding)
        # similarities[image_name] = np.linalg.norm(image_embedding-embedding)
    # return closest image
    closest_img = list(similarities.keys())[0]
    min_dist = similarities[closest_img]
    for img_name in similarities:
        if similarities[img_name] < min_dist:
            closest_img = img_name
    image = Image.open(images_dir+closest_img)
    return image

st.title("Image Retrieval from Text")
st.write("Note: Our model hasn't been trained sufficiently and \
the results are nowhere close to our expectations. \
    We'll be improving the model as we find time and more GPU resources. \
        Until then, play around with this (not so great) model.")

selectbox = st.sidebar.multiselect(
        "Select a maximum of two captions",
        list(model.wv.vocab.keys())
        )
if len(selectbox) == 2:
    operation = st.sidebar.radio(label="Select operation", options=["Add", "Subtract"])
    if operation == "Add":
        sign = "+"
        positive_inputs, negative_inputs = selectbox, []
        message = f"Closest image to '{positive_inputs[0]}{sign}{positive_inputs[1]}'"
    elif operation == "Subtract":
        sign = "-"
        positive_inputs, negative_inputs = [selectbox[0]], [selectbox[1]]
        message = f"Closest image to '{positive_inputs[0]}{sign}{negative_inputs[0]}'"
elif len(selectbox) == 1:
    positive_inputs, negative_inputs = selectbox, []
    message = f"Closest image to '{positive_inputs[0]}'"

search = st.sidebar.button("Search")

if search and len(selectbox) <= 2:
    img = get_closest_image(positive_inputs, negative_inputs)
    st.image(img)
    st.write(message)
else:
    st.write("Please select a maximum of two captions.")
