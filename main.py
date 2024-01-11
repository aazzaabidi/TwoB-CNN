import sys
import os
import argparse
from pathlib import Path
import numpy as np
from utils import format_X_pxl, format_X_obj, format_y
from train import run
from test import restore
from models import OneBranchModel, TwoBranchModel

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    
    # Parsing arguments
    if len(sys.argv) == 1:
        print ('Usage: python '+sys.argv[0]+' data_path num_split [options]' )
        print ('Help: python '+sys.argv[0]+' -h/--help')
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',help='Path to data',type=str)
    parser.add_argument('num_split',help='Number of split to use',type=str)
    parser.add_argument('-m','--model',help='Which model to execute',choices=['1D','2D','3D','TempCNN'],default='1D',type=str)
    parser.add_argument('-l','--level',help='Which level to leverage: pixel with/without object',choices=['pxl','pxl-obj','concat'],default='pxl',type=str)
    parser.add_argument('-s','--stats',help='Which statistics to use for object data',nargs='+',choices=['mean','median'],default=['mean'])
    parser.add_argument('-w','--weight',help='Weighting of auxiliary classifiers',default=0.5,type=float)
    parser.add_argument('-out','--out_path',help='Output path for model and results',type=str)
    parser.add_argument('-bs','--batch_size',dest='batch_size',help='Batch size',default=256,type=int)
    parser.add_argument('-ep','--num_epochs',dest='num_epochs',help='Number of training epochs',default=100,type=int)
    parser.add_argument('-lr','--learning_rate',dest='learning_rate',help='Learning rate',default=1e-4,type=float)
    parser.add_argument('-tqdm',dest='tqdm',help='Display tqdm progress bar',default=False,type=boolean_string)
    args = parser.parse_args()

    # Get argument values
    data_path = args.data_path
    n_split = args.num_split
    model_to_train = args.model
    level = args.level
    weight = args.weight
    obj_stats = args.stats

    if level in ['pxl-obj','concat']:
        try:
            assert model_to_train not in ['2D','3D']
        except AssertionError :
            print ("Can't leverage Pixel and Object levels data with 2D/3D-CNN")

    if not args.out_path is None :
        out_path = args.out_path
    else:
        out_path = f'CNN_{model_to_train}' if model_to_train != 'TempCNN' else  model_to_train
        if model_to_train in ['1D','TempCNN']:
            out_path += f'_{level}'
    batch_size = args.batch_size
    n_epochs = args.num_epochs
    lr = args.learning_rate
    tqdm_display = args.tqdm
    
    # Create output path if does not exist
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # Load Training and Validation set
    train_y = format_y(os.path.join(data_path,f'train_gt_{n_split}.npy') )
    print ('Training GT:',train_y.shape)
    valid_y = format_y(os.path.join(data_path,f'valid_gt_{n_split}.npy') )
    print ('Validation GT:', valid_y.shape)
    n_classes = len(np.unique(train_y))
    print ('Number of classes:',n_classes)

    train_X_obj,valid_X_obj = (None,None)

    if level.startswith('pxl'):
        train_X_pxl = format_X_pxl (os.path.join(data_path,f'train_X_pxl_{n_split}.npy'), model_to_train )
        print ('Training X Pixel:',train_X_pxl.shape)
        valid_X_pxl = format_X_pxl (os.path.join(data_path,f'valid_X_pxl_{n_split}.npy'), model_to_train )
        print ('Validation X Pixel:',valid_X_pxl.shape)

    if level == 'pxl-obj':
        train_X_obj = format_X_obj (os.path.join(data_path,f'train_X_mean_{n_split}.npy'),
                                    os.path.join(data_path,f'train_X_median_{n_split}.npy'), obj_stats )
        print ('Training X Object:',train_X_obj.shape)
        valid_X_obj = format_X_obj (os.path.join(data_path,f'valid_X_mean_{n_split}.npy'),
                                    os.path.join(data_path,f'valid_X_median_{n_split}.npy'), obj_stats )
        print ('Validation X Object:',valid_X_obj.shape)

    elif level == 'concat':
        train_X_pxl = np.concatenate ((format_X_pxl (os.path.join(data_path,f'train_X_pxl_{n_split}.npy'), model_to_train ),
                                      format_X_obj (os.path.join(data_path,f'train_X_mean_{n_split}.npy'),
                                                    os.path.join(data_path,f'train_X_median_{n_split}.npy'), obj_stats )), axis=-1)
        print ('Training X Concat:',train_X_pxl.shape)
        valid_X_pxl = np.concatenate ((format_X_pxl (os.path.join(data_path,f'valid_X_pxl_{n_split}.npy'), model_to_train ),
                                      format_X_obj (os.path.join(data_path,f'valid_X_mean_{n_split}.npy'),
                                                    os.path.join(data_path,f'valid_X_median_{n_split}.npy'), obj_stats )), axis=-1)
        print ('Validation X Concat:',valid_X_pxl.shape)

    # Create the object model

    if level in ['pxl','concat']:
        model = OneBranchModel (n_classes,encoder=model_to_train)
    elif level == 'pxl-obj':
        model = TwoBranchModel (n_classes,encoder=model_to_train)

    # Learning stage
    checkpoint_path = os.path.join(out_path,f'model_{n_split}')
    run (model,train_X_pxl,train_X_obj,train_y,valid_X_pxl,valid_X_obj,valid_y,
         checkpoint_path,batch_size,lr,n_epochs,weight,level,tqdm_display)

    # Load Test set 
    test_y = format_y (os.path.join(data_path,f'test_gt_{n_split}.npy'), encode=False )
    print ('Test GT:',test_y.shape)

    test_X_obj = None

    if level.startswith('pxl'):
        test_X_pxl = format_X_pxl (os.path.join(data_path,f'test_X_pxl_{n_split}.npy'), model_to_train )
        print ('Test X Pixel:',test_X_pxl.shape)

    if level == 'pxl-obj':
        test_X_obj = format_X_obj (os.path.join(data_path,f'test_X_mean_{n_split}.npy'),
                                    os.path.join(data_path,f'test_X_median_{n_split}.npy'), obj_stats )
        print ('Test X Object:',test_X_obj.shape)
    
    elif level == 'concat':
        test_X_pxl = np.concatenate ((format_X_pxl (os.path.join(data_path,f'test_X_pxl_{n_split}.npy'), model_to_train ),
                                      format_X_obj (os.path.join(data_path,f'test_X_mean_{n_split}.npy'),
                                                    os.path.join(data_path,f'test_X_median_{n_split}.npy'), obj_stats )), axis=-1)
        print ('Test X Concat:',test_X_pxl.shape)
    

    # Inference stage
    result_path = os.path.join(out_path,f'pred_{n_split}.npy')
    restore (model,test_X_pxl,test_X_obj,test_y,batch_size,checkpoint_path,result_path,level,tqdm_display)
