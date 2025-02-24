#!/bin/bash

# List of model names
models=("resnet50" "vgg16" "resnet152" "vgg19")
lrs=("0.001")
optims=("adam" "radam" "sgd")
batch_sizes=("64" "256")

# Loop through the models and run the Python script


# testing different optimizers in vgg16
for bs in "${batch_sizes[@]}" 
do
    echo "Training model vgg16 [optim=adaman, bs=$bs, lr=0.001]"
    python main.py --name="vgg16_adam__0.001_${bs}" --group="vgg16_v2" --model="vgg16" --epochs=10 --lr=0.001 --optim=adam --batch_size=$bs
done



# # testing different models
# for model in "${models[@]}"
# do
#     echo "Training model: $model"
#     python main.py --name="$model" --group="models" --model="$model" --epochs=30 --lr=0.001
# done

