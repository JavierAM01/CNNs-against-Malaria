#!/bin/bash

# List of model names
models=("resnet50" "vgg16" "resnet152" "vgg19")
lrs=("0.01" "0.001")
optims=("adam" "radam" "sgd")

# Loop through the models and run the Python script


# testing different optimizers in vgg16
for optim in "${optims[@]}" 
do
    for lr in "${lrs[@]}"
    do
        echo "Training model vgg16 [optim=$optim, lr=$lr]"
        python main.py --name="vgg16_$optim_$lr" --gruop="vgg16" --model="vgg16" --epochs=10 --lr=$lr --optim=$optim
    done
done



# testing different models
for model in "${models[@]}"
do
    echo "Training model: $model"
    python main.py --name="$model" --gruop="models" --model="$model" --epochs=30 --lr=0.001
done

