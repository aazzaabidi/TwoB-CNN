export CUDA_VISIBLE_DEVICES=3

data_path='../splits'

for i in {3..5}
do 
    python main.py $data_path $i -m 2D
done

for i in {2..5}
do 
    python main.py $data_path $i -m 3D
done