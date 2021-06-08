import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split


def train_test_split(df, index,
                     split_by=None,
                     stratify_by=None,
                     test_split=0.1,
                     seed=None):
    """Split pandas dataframe into train and test splits. Options to split by
    group (i.e. keep groups together) and stratify by label.

    Args:
        df (pandas dataframe): dataframe to split. Must contain columns index as well as
                     split_by and stratify_by (if not None)
        index (string): name of column that acts as index to dataset
        split_by (str, optional): column to group together and split by. Defaults to None.
        stratify_by (str, optional): column to stratify by. Defaults to None.
        test_split (float, optional): test proportion. Defaults to 0.1.
        seed (int, optional): random seed can be fixed for reproducible splits. 
                              Defaults to None.
    """
    
    if index not in df:
        raise ValueError(f'{index} not in df')
        
    if split_by is None:
        split_by = index
    elif split_by not in df:
        raise ValueError(f'{split_by} not a column in df')

    df_unique = df.drop_duplicates(subset=[split_by])

    if stratify_by is None:
        stratify = None
    else:
        if stratify_by in df:
            stratify = df_unique[stratify_by]
        else:
            raise ValueError(f'{stratify_by} not a column in df')
 
    train_ids, test_ids = sklearn_split(df_unique[split_by], 
                                           test_size=test_split,
                                           stratify=stratify,
                                           random_state=seed)
    
    df_train = df.set_index(split_by).drop(test_ids).reset_index()
    df_test = df.set_index(split_by).drop(train_ids).reset_index()

    return df_train, df_test


if __name__ == '__main__':
    
    labels_csv = '/home/lkoch/data/kaggle/trainLabels.csv'
    df = pd.read_csv(labels_csv)

    df['patient_id'] = df.apply(lambda row: row['image'].split('_')[0], axis=1)
    df['side'] = df.apply(lambda row: row['image'].split('_')[1], axis=1)
    train, test = train_test_split(df, 'image', split_by='patient_id')

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

    
    
    
    print('done.')