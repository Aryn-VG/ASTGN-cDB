# AadaptiveS-tgn

## Running the experiments

### dataset and preprocessing

#### Data downloading

* [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
* [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

#### Data preprocessing

We use the dense `npy` format to save the features in binary format. If edge features or nodes features are absent, it will be replaced by a vector of zeros. 

* Step1 :

```{bash}
python preprocess_csv.py  --data wikipedia/reddit
```

* Step2:

```python
python BuildDglGraph.py --data wikipedia/reddit
```

#### Runing the model
```{bash}
python train.py  --data wikipedia/reddit/mooc/uci/socialevolve/enron
```

### Requirements

* python>=3.7

* Dependency

  ```python
  dgl==0.5.2
  numpy==1.18.1
  scikit-learn==0.23.2
  torch==1.5.0
  tqdm==4.48.2
  ```



#### General flag

Optional arguments are described in args.py.



