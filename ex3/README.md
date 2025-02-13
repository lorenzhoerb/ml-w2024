# Exercise 3: 3.1.1.4 Model Stealing / Extraction

## Installation

### Environment

- Python 3

1. Create a python virtual environment

```sh
python3 -m venv venv
```

2. Activate virtual environment

- Mac / Linux

```sh
source venv/bin/activate
```

- Windows

```sh
venv\Scripts\activate
```

3. Install requirements

```sh
pip install -r requirements.txt
```

## Datasets

You will need six datasets to perform all experiments in the paper, all extracted into the data/ directory.

- **New Datasets**
    - pokemon-dataset-1000 ([Link](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000). Images in `data/pokemon-dataset-1000/`)
    - emotion-recognition-dataset ([Link](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset). Images in `data/emotion-recognition-dataset`)
- **Dataset used in Paper (for validation)**
    - CUB-200-2011 Bird Dataset ([Link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Images in `data/CUB_200_2011/images/<classname>/*.jpg`)



