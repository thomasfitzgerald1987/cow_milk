import torch
from sklearn.model_selection import train_test_split
from functions import *

def load_dataset_from_directory(output_dir, test_prop=.20, valid_prop = .20, make_copy=False):
    """
    :param output_dir: Directory to save output to, if make_copy=True.
    :param test_prop: Proportion of dataset to reserve as test set.
    :param valid_prop: Proportion of dataset to use as validation set.
    :param make_copy: Saves a copy of train/validation/test sets if set to True.
    :return: x and y training ,validation, and test sets.
    """
    #Import
    x_data = pd.read_csv(os.path.join(output_dir,'x_data.csv'))
    y_data = pd.read_csv(os.path.join(output_dir,'y_data.csv'))
    #Cleanup
    x_data = x_data.drop(list(x_data.filter(regex='Unnamed')), axis=1)
    y_data = y_data.drop(list(y_data.filter(regex='Unnamed')), axis=1).drop(['Index'],axis=1)
    #To numpy array
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()
    #Convert to tensor
    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)
    #Replace all nan with 0
    x_data = torch.nan_to_num(x_data,nan=0.0)
    y_data = torch.nan_to_num(y_data,nan=0.0)

    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=test_prop,shuffle=True)
    valid_prop = abs(valid_prop/(test_prop-1))
    x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=valid_prop,shuffle=True)

    if make_copy:
        pd.DataFrame(x_train.numpy()).to_csv(os.path.join(output_dir, 'x_train.csv'))
        pd.DataFrame(x_valid.numpy()).to_csv(os.path.join(output_dir, 'x_valid.csv'))
        pd.DataFrame(x_test.numpy()).to_csv(os.path.join(output_dir, 'x_test.csv'))
        pd.DataFrame(y_train.numpy()).to_csv(os.path.join(output_dir, 'y_train.csv'))
        pd.DataFrame(y_valid.numpy()).to_csv(os.path.join(output_dir, 'y_valid.csv'))
        pd.DataFrame(y_test.numpy()).to_csv(os.path.join(output_dir, 'y_test.csv'))
    return(x_train,x_valid,x_test,y_train,y_valid,y_test)


def load_full_dataset(output_dir):
    """
    :param output_dir: Directory to save output to, if make_copy=True.
    :param test_prop: Proportion of dataset to reserve as test set.
    :param valid_prop: Proportion of dataset to use as validation set.
    :param make_copy: Saves a copy of train/validation/test sets if set to True.
    :return: x and y training ,validation, and test sets.
    """
    #Import
    x_data = pd.read_csv(os.path.join(output_dir,'x_data.csv'))
    y_data = pd.read_csv(os.path.join(output_dir,'y_data.csv'))
    #Cleanup
    x_data = x_data.drop(list(x_data.filter(regex='Unnamed')), axis=1)
    y_data = y_data.drop(list(y_data.filter(regex='Unnamed')), axis=1).drop(['Index'],axis=1)
    #To numpy array
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()
    #Convert to tensor
    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)
    #Replace all nan with 0
    x_data = torch.nan_to_num(x_data,nan=0.0)
    y_data = torch.nan_to_num(y_data,nan=0.0)

    return(x_data,y_data)