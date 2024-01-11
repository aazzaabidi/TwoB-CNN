import numpy as np
from sklearn.preprocessing import LabelEncoder

def format_X_pxl (array_path,model_to_train):
    '''
    Format training data
    output shape: if 3D (number of samples, width, height, timestamps, bands)
                  elif 2D (number of samples, width, height, timestamps * bands)
                  elif 1D (number of samples, timestamps, bands)
    '''
    if model_to_train in['1D','1D_Concat','1D_Sum'] or model_to_train == 'TempCNN':
        array = np.load(array_path)
        assert array.shape[1] == array.shape[2]
        center = (array.shape[1]-1)//2
        array = array[:,center,center,:,:]
    elif model_to_train == '2D':
        array = np.load(array_path)
        array = array.reshape(array.shape[0],array.shape[1],array.shape[2],-1)
    elif model_to_train == '3D':
        array = np.load(array_path)
    return array

def format_X_obj (arrayMean_path,arrayMedian_path,stats):
    if len(stats) > 1 :
        array = np.concatenate([np.load(arrayMean_path),np.load(arrayMedian_path)],axis=-1)
    elif stats[0] == 'mean':
        array = np.load(arrayMean_path)
    elif stats[0] == 'median':
        array = np.load(arrayMedian_path)
    return array    

def format_y (array_path,encode=True):
    '''
    Format ground truth data
    Encode label (second column) with values between 0 and n_classes-1.
    output shape: (number of samples,)
    '''
    array = np.load(array_path)[:,1]
    if encode :
        encoder = LabelEncoder()
        array = encoder.fit_transform( array )
    return array

def transform_y (y,prediction):
    '''
    Transform labels back to original encoding
    output shape: (number of samples,)
    '''
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder.inverse_transform(prediction)

def get_iteration (array, batch_size):
    '''
    Function to get the number of iterations over one epoch w.r.t batch size
    '''
    n_batch = int(array.shape[0]/batch_size)
    if array.shape[0] % batch_size != 0:
        n_batch+=1
    return n_batch

def get_batch (array, i, batch_size):
    '''
    Function to select batch of training/validation/test set
    '''
    start_id = i*batch_size
    end_id = min((i+1) * batch_size, array.shape[0])
    batch = array[start_id:end_id]
    return batch.astype(np.float32)