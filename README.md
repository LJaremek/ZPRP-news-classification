zprp
==============================

ZPRP project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict.py
    │   │   └── train.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

### Build start environment
#### Short version without explanations
```shell
conda create --name zprp python=3.10.9
conda activate zprp
conda install --file requirements.txt
```

#### Create conda environment
```shell
conda create --name zprp python=3.10.9
```

#### Activate conda environment
```shell
conda activate zprp
```

#### Install requirements
```shell
conda install --file requirements.txt
```

#### How to create `requirements.txt`
```shell
conda list -e > requirements.txt
```
or
```shell
pip list --format=freeze > requirements.txt
```


### Run project
#### Run tests
```shell
python -m unittest
```

#### Run server with endpoints
```shell
python endpoints.py
```


#### Dataset
The **ISOT Fake News dataset** is a compilation of several thousands fake news and truthful articles, obtained from different legitimate news sites and sites flagged as unreliable by Politifact.com.

Description:
https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf

Download:
https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip

#### Train LSTM
Firstly it is necessary to set adequate data path and optionally change hiperparameters in `src/config`, then create directories `models/checkpoints`, `models/pickles` and `models/log` and then run 
```shell
python3 /src/features/build_features.py
```
which should generate pickles in `models/pickles` of embeddings, tokenized articles and corpus. Later you can start traning 
by running
```shell
python3 /src/model/train.py
```

All the logs and plots are saved in `models/log`, and checkpoints are saved in `models/checkpoints`

### Evaluation LSTM
To evaluate LSTM you should firstly set paths in `src/models/evaluate.py`. Make sure that values in config match with embeddings
that pretrained model was trained on.
```shell
python3 /src/model/evaluate.py
```

### Predict LSTM
To predict set TXT value that you want to make prediction, set path to model and then run
```shell
python3 /src/model/predict.py
```

#### Project structure
Project folder:
* data:
    * Fake.csv
    * True.csv
