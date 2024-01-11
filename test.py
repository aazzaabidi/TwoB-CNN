import time
import numpy as np 
import tensorflow as tf 
from utils import get_batch, get_iteration, transform_y
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score, confusion_matrix ,classification_report
from tqdm import tqdm



def predict_by_batch (model,test_X_pxl,test_X_obj,test_y,batch_size,level,tqdm_display): 
    '''
    Predict batch of test set
    '''
    pred=[]

    iteration = get_iteration(test_y,batch_size)
    if not tqdm_display:
        print (f'Test batchs: {iteration}')

    start = time.time()
    if level in ['pxl','concat']:
        for batch in tqdm(range(iteration),disable=not(tqdm_display)):
            batch_X = get_batch (test_X_pxl,batch,batch_size)
            batch_pred = model(batch_X,is_training = False) 
            pred.append(tf.argmax(batch_pred,axis=1))
            del batch_X,batch_pred
    elif level == 'pxl-obj':
        for batch in tqdm(range(iteration),disable=not(tqdm_display)):
            batch_X_pxl = get_batch (test_X_pxl,batch,batch_size)
            batch_X_obj = get_batch (test_X_obj,batch,batch_size)
            batch_pred,_,_ = model(batch_X_pxl,batch_X_obj,is_training = False) 
            pred.append(tf.argmax(batch_pred,axis=1))
            del batch_X_pxl,batch_X_obj,batch_pred
    stop = time.time()
    elapsed = stop - start

    pred = np.hstack(pred)
    return pred, elapsed 
        
def restore (model,test_X_pxl,test_X_obj,test_y,batch_size,checkpoint_path,result_path,level,tqdm_display):
    '''
    Load weights for best configuration and evaluate on test set
    '''
    model.load_weights(checkpoint_path)
    print ('Weights loaded')

    pred, elapsed = predict_by_batch (model,test_X_pxl,test_X_obj,test_y,batch_size,level,tqdm_display)
    if not tqdm_display:
        print (f'Test Time: {elapsed}')
    pred = transform_y (test_y,pred)
    np.save (result_path,pred)

    print ('Acc:',accuracy_score(test_y,pred))
    print ('F1:',f1_score(test_y,pred,average='weighted'))
    print ('Kappa:',cohen_kappa_score(test_y,pred))
