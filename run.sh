#!/bin/bash

# List of model names
models=("resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "vgg16" "vgg19" "densenet121" "mobilenet_v2" "inception_v3")

# Loop through the models and run the Python script
for model in "${models[@]}"
do
    echo "Training model: $model"
    python main.py "$model"
done

