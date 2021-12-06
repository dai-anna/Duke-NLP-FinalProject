# Duke NLP Final Project <img width=90 align="right" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Duke_University_logo.svg/1024px-Duke_University_logo.svg.png">

We collect data on seven different topics on twitter to build an LDA model as well as a discriminative Neural Network to benchmark their perfomance in topic classification.

## Hashtags
1) crypto
2) tesla
3) goldenstatewarriors
4) formula1
5) thanksgiving
6) holidays
7) covid19

## Steps to Reproduce
1) Create & activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
2) Install dependencies
```bash
make install
```
3) Collect data (takes long, please skip and use content in `data/`) 
```bash
make data-collect
```
4) Train LDA model
TBD

5) Train Neural Network
TBD
