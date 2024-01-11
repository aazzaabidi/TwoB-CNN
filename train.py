import time
import numpy as np 
import tensorflow as tf 
from utils import get_batch, get_iteration
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from tqdm import tqdm
import keras

def train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred):
    '''
    Output of training step
    Save model if accuracy improves
    '''
    print (f'Epoch {epoch+1}, Loss: {train_loss.result()}, Acc: {train_acc.result()}, Valid Loss: {valid_loss.result()}, Valid Acc: {valid_acc.result()}, Time: {elapsed}')
    if valid_acc.result() > best_acc :
        print ( f1_score (valid_y,pred,average=None) )
        model.save_weights(checkpoint_path)
        print (f'{valid_acc.name} improved from {best_acc} to {valid_acc.result()}, saving to {checkpoint_path}')
        best_acc = valid_acc.result()
            
    # Reset metrics for the next epoch
    train_loss.reset_states()
    train_acc.reset_states()
    valid_loss.reset_states()
    valid_acc.reset_states()
    
    return best_acc

@tf.function
def train_step (model, x_pxl, x_obj, y, loss_function, optimizer, loss, metric, weight, level, is_training):
    '''
    Gradient differentiation
    '''
    with tf.GradientTape() as tape:
        if level in ['pxl','concat']:
            pred = model(x_pxl,is_training)
            cost = loss_function(y,pred)
        elif level == 'pxl-obj':
            pred, pxl_pred, obj_pred = model(x_pxl,x_obj,is_training)
            cost = loss_function(y,pred)
            cost += weight * loss_function(y,pxl_pred) 
            cost += weight * loss_function(y,obj_pred)

        if is_training :
            gradients = tape.gradient(cost, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss(cost)
        metric(y, tf.math.argmax(pred,axis=1))
    return  tf.math.argmax(pred,axis=1)

def run (model,train_X_pxl,train_X_obj,train_y,
            valid_X_pxl,valid_X_obj,valid_y,checkpoint_path,
            batch_size,lr,n_epochs,weight,level,tqdm_display) :
    '''
    Main function for training models
    '''
    
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.Accuracy(name='train_acc')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.Accuracy(name='valid_acc')
    
    best_acc = float("-inf")

    train_iter = get_iteration (train_y,batch_size)
    valid_iter = get_iteration (valid_y,batch_size)
    if not tqdm_display:
        print (f'Training batchs: {train_iter}')
        print (f'Validation batchs: {valid_iter}')
    
    if level in ['pxl','concat']:
        for epoch in range(n_epochs):
            start = time.time()
            train_X_pxl, train_y = shuffle(train_X_pxl, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_X = get_batch (train_X_pxl,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_X,None,batch_y,loss_function,optimizer,train_loss,train_acc,weight,level,is_training=True)
                del batch_X,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_X = get_batch (valid_X_pxl,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_X,None,batch_y,loss_function,optimizer,valid_loss,valid_acc,weight,level,is_training=False)
                pred.append(batch_pred)
                del batch_X,batch_y,batch_pred
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if epoch == 0:
                print (model.summary())
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
    
    elif level =='pxl-obj':
        for epoch in range(n_epochs):
            start = time.time()
            train_X_pxl, train_X_obj, train_y = shuffle(train_X_pxl, train_X_obj, train_y, random_state=0)
            for batch in tqdm(range(train_iter),disable=not(tqdm_display)):
                batch_X_pxl = get_batch (train_X_pxl,batch,batch_size)
                batch_X_obj = get_batch (train_X_obj,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_X_pxl,batch_X_obj,batch_y,loss_function,optimizer,train_loss,train_acc,weight,level,is_training=True)
                del batch_X_pxl,batch_X_obj,batch_y
            pred = []
            for batch in tqdm(range(valid_iter),disable=not(tqdm_display)):
                batch_X_pxl = get_batch (valid_X_pxl,batch,batch_size)
                batch_X_obj = get_batch (valid_X_obj,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_X_pxl,batch_X_obj,batch_y,loss_function,optimizer,valid_loss,valid_acc,weight,level,is_training=False)
                pred.append(batch_pred)
                del batch_X_pxl,batch_X_obj,batch_y,batch_pred
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            if epoch == 0:
                print (model.summary())
                #tf.keras.utils.plot_model(model, to_file='/media/DATA/AZZA/2D_3D_CNN/2branches', show_shapes=True)

            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
