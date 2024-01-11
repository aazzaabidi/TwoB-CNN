# TwoB-CNN

# Code Repository Readme

## Multivariate Satellite Image Time Series (SITS) Classification

### Overview
This repository contains code for multivariate SITS classification using various models and levels of feature extraction. The code facilitates training, validation, and testing of the models, allowing for experimentation with different configurations.

### Requirements
- Python 3.x
- NumPy
- TensorFlow
- Matplotlib
- tqdm

### Usage

#### Training and Evaluation
To train and evaluate the models, run the following command:

```bash
python main.py data_path num_split [options]
```

- `data_path`: Path to the dataset.
- `num_split`: Number of splits to use for cross-validation.

Optional arguments:
- `-m, --model`: Model to execute (choices: '1D', '2D', '3D', 'TempCNN', default: '1D').
- `-l, --level`: Level to leverage: 'pxl' (pixel), 'pxl-obj' (pixel and object), 'concat' (concatenated, default: 'pxl').
- `-s, --stats`: Statistics to use for object data (choices: 'mean', 'median', default: 'mean').
- `-w, --weight`: Weighting of auxiliary classifiers (default: 0.5).
- `-out, --out_path`: Output path for model and results.
- `-bs, --batch_size`: Batch size (default: 256).
- `-ep, --num_epochs`: Number of training epochs (default: 100).
- `-lr, --learning_rate`: Learning rate (default: 1e-4).
- `-tqdm`: Display tqdm progress bar (default: False).

#### Example
```bash
python main.py /path/to/dataset 1 -m 1D -l pxl -s mean -w 0.5 -out /output/path -bs 256 -ep 100 -lr 1e-4 -tqdm
```

### Directory Structure
- `utils.py`: Utility functions for data formatting.
- `train.py`: Training script for the models.
- `test.py`: Evaluation script for the models.
- `models.py`: Model definitions (OneBranchModel, TwoBranchModel).
- `main.py`: Main script to execute training and evaluation.

### Output
The trained models and evaluation results are stored in the specified output path.

### License
This code is released under the MIT License.

### Citation
If you use this code in your research, please cite our paper     @article{Abidi2022,
    author = {A. Abidi, A. Ben Abbes, Y. J. E. Gbodjo, D. Ienco and I. R. Farah},
    title = {Combining pixel- and object-level information for land-cover mapping using time-series of Sentinel-2 satellite data},
    journal = {Remote Sensing Letters},
    volume = {13},
    number = {2},
    pages = {162-172},
    year = {2022},
    publisher = {Taylor & Francis},
    doi = {10.1080/2150704X.2021.2001071},
    URL = { https://doi.org/10.1080/2150704X.2021.2001071},
    eprint = { https://doi.org/10.1080/2150704X.2021.2001071}
}

