# Stock Prediction Framework 
This framework is built upon multi-GPU DDP (Distributed Data Parallel), utilizing minute-level factors to predict stock price movements.

## Algorithms
There are already four algorithms in this framwork (Vanilla, iTransformer, deeplob, GRUModel), you can use different algorithms in configs/config.py

_Sorry, due to confidentiality constraints, I am unable to provide the most efficacious algorithm._
### Vanilla
traditional transformer structure

### GRU
GRU framework model

## Data
_Sorry we cannot provide any data, because the data we use have private factors_

We utilize minute-level stock data, employing various factors as features, to predict the rise and fall of stock prices. You can make your own data.

## Training Framework
Leveraging Distributed Data Parallel (DDP) Architecture for Model Training
