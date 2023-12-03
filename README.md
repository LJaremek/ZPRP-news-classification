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


### Dataset
The **ISOT Fake News dataset** is a compilation of several thousands fake news and truthful articles, obtained from different legitimate news sites and sites flagged as unreliable by Politifact.com.

Description:
https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf

Download:
https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip

### Project structure
Project folder:
* data:
    * Fake.csv
    * True.csv
