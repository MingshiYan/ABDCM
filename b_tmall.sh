#!/bin/bash


# lr=('0.001' '0.0001')
lr=('0.0001')
emb_size=(64)
reg_weight=('0.0001')
# lambdas=('0.0' '0.1' '0.3' '0.5' '0.7' '0.9' '1.0' '1.1' '1.2' '1.5')
lambdas=('0.5')

# behaviors=("['collect', 'cart', 'buy']" "['click', 'cart', 'buy']" "['click', 'collect', 'buy']" "['click', 'buy']" "['collect', 'buy']" "['cart', 'buy']")
behaviors=("['click', 'collect', 'cart', 'buy']")
layers=(2')

dataset=('tmall')
device='cuda:5'
batch_size=1024
decay=0

data_loader='data_set'
model_name='model'
log_name='model'
gpu_no=1

for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
            for lamb in ${lambdas[@]}
            do
            for layer in ${layers[@]}
            do
            for bhv in "${behaviors[@]}"
            do
                echo 'start train: '$name
                `
                    python main.py \
                        --model_name $model_name \
                        --log_name $log_name \
                        --data_name $name \
                        --lr ${l} \
                        --lamb ${lamb} \
                        --layers ${layer} \
                        --behaviors "${bhv}" \
                        --gpu_no $gpu_no \
                        --reg_weight ${reg} \
                        --embedding_size $emb \
                        --device $device \
                        --decay $decay \
                        --data_loader $data_loader \
                        --batch_size $batch_size 
                              
                `
                echo 'train end: '$name
            done
            done
            done
            done
        done
    done
done