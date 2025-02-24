#!/bin/bash

# List of model names
models=("resnet50" "vgg16" "resnet152" "vgg19")

# Loop through the models and run the Python script
for model in "${models[@]}"
do
    echo "Training model: $model"
    python main.py "dataagumentation/$model"
done

