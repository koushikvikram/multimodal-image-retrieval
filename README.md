# Multimodal Image Retrieval

[![Pylint](https://github.com/koushikvikram/multimodal-image-retrieval/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/koushikvikram/multimodal-image-retrieval/actions/workflows/pylint.yml)  [![Pytest](https://github.com/koushikvikram/multimodal-image-retrieval/actions/workflows/pytest.yml/badge.svg)](https://github.com/koushikvikram/multimodal-image-retrieval/actions/workflows/pytest.yml)

> 🚦⚠️👷‍♂️🏗️ Repo Under Construction 🚦⚠️👷‍♂️🏗️

A deep learning application to retrieve images by searching with text

## Dataset

Download the InstaNY100K dataset from this [Google Drive link](https://drive.google.com/file/d/1blGgEOlrHrM0-NAQxYVRwMlfiHDvVHXb/view?usp=sharing)

Extract the dataset in the path, `./datasets/raw/`. You folder structure should look like the one below:

```
./datasets/raw/
|
|-- InstaNY100K
    |
    |-- captions
    |   |
    |   |-- newyork
    |      | 1487768220566960691.txt
    |      | 1490727714071958379.txt
    |      | ...
    |   
    |-- img_resized
        |
        |-- newyork
            | 1480879485913200243.jpg
            | 1480879539524935620.jpg
            | ...
```


## GitHub Actions for this Repository

[Pylint](https://github.com/koushikvikram/multimodal-image-retrieval/blob/main/.github/workflows/pylint.yml) - Code Quality Check

## Exploring the Word2Vec Model

<p align="middle">
  <img src="images/similar/man.png" height=250 width="250" />
  <img src="images/similar/family.png" height=250 width="250" /> 
  <img src="images/similar/healthy.png" height=250 width=250" />
  <img src="images/similar/car.png" height=250 width=250" />
  <img src="images/similar/dog.png" height=250 width=250" />
  <img src="images/similar/sports.png" height=250 width=250" />
  <img src="images/similar/hairstyle.png" height=250 width=250" />
  <img src="images/similar/food.png" height=250 width=250" />
  <img src="images/similar/toronto.png" height=250 width=250" />
</p>

