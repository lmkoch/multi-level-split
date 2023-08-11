# Multilevel Train Test Splits

Small wrapper around sklearn's `train_test_split` to allow splitting by higher-order index. This is necessary to prevent train-test leaks when multiple data points exist per group that should not be split, e.g. multiple images per patient.

## Install

````
> pip install git+https://github.com/lmkoch/multi-level-split
````

## Use source

In case you want to work with the source code, you should install the dependencies `pandas` and `sklearn`, e.g. with pipenv and the Pipfile provided

````
project_root> pipenv install
````

## Example usage

The function `train_test_split` splits a  pandas dataframe into train and test splits. It returns a train and test pandas dataframe.

````python

from multi_level_split.util import train_test_split
import pandas as pd

# dummy dataset
num_groups = 20
members_per_group = 10
class_imbalance = 0.25
groups = list(range(num_groups)) * members_per_group
idx = list(range(num_groups * members_per_group))
labels = [ele < num_groups * class_imbalance for ele in groups]

df = pd.DataFrame({'index': idx, 'group': groups, 'label': labels})
# shuffle it
df = df.sample(frac = 1)

# split by group, stratified by label
train, test = train_test_split(df, 'index', 
                                split_by='group', 
                                stratify_by='label',
                                test_split=0.2)
````

