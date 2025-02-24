#!/bin/bash

# List of model names
models=("resnet18" "resnet50" "resnet152" "vgg16" "vgg19")

# Loop through the models and run the Python script
for model in "nofreeze/${models[@]}"
do
    echo "Training model: $model"
    python main.py "$model"
done

