# Duke NLP Final Project <img width=90 align="right" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Duke_University_logo.svg/1024px-Duke_University_logo.svg.png">

We collect data on seven different topics on twitter to build an LDA model as well as a discriminative Neural Network to benchmark their performance in topic classification.

Read our final report [here](https://github.com/dai-anna/Duke-NLP-FinalProject/blob/884f2b059f1ed0d85f596d90068489be4c03bcec/report/report_submission_flat.pdf).


## Hashtags
1) crypto
2) tesla
3) championsleague
4) formula1
5) thanksgiving
6) holidays
7) covid19

## Steps to Reproduce
#### 1) Create & activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2) Install dependencies
```bash
make install
```

#### 3) Collect data (takes long, please skip and use content in `data/`) 
```bash
make data-collect
```

#### 4) Train LDA model
```bash
cd src
python3 lda_modeling.py
```


#### 5) Train Neural Network (Requires multiple hours on a GPU enabled device)
```bash
cd src
# tune hyperparameters
python3 tf_hyperparameter_tuning.py
```

Visually inspect results in `visualize_study.ipynb`

```bash
# run neural network with chosen params
cd src
python3 tf_train_model_with_best_params.py
```


## Contributors

| Name | Reference |
|---- | ----|
|Anna Dai | [GitHub Profile](https://github.com/dai-anna)|
|Satvik Kishore| [GitHub Profile](https://github.com/satvikk)|
|Moritz Wilksch |[GitHub Profile](https://github.com/moritzwilksch)|


