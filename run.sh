#!/bin/bash

# List of model names
# models=("resnet50" "vgg16" "resnet152" "vgg19")
models=("vgg16" "densenet121" "mobilenet_v3" "efficientnet_b0" "regnet_y_400mf" "convnext_tiny")
lrs=("0.001")
optims=("adam" "radam" "sgd")
batch_sizes=("64" "256")

# Loop through the models and run the Python script


# testing different optimizers in vgg16
for model in "${models[@]}" 
do
    echo "Training model ${model} [optim=adamn, bs=128, lr=0.001]"
    python main.py --name="${model}" --group="test_models" --model=$model --epochs=5 --lr=0.001 --optim=adam --batch_size=128
done
