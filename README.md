# AS homework

## Installation guide

### Creating virtual enviroment
This project is using [poetry](https://python-poetry.org/) to manage dependencies. To use this project you need to [install](https://python-poetry.org/docs/) it.  
After installing poetry, run
```shell 
poetry install
```
This command will create virtual enviroment. You can either enter it using
```shell 
poetry shell
```
or start all commands with poetry
```shell 
poetry run python train.py
```

Note:
If you want to use CUDA with this project, you need to [install](https://developer.nvidia.com/cuda-11-8-0-download-archive) it separately. The supported version is 11.8.

### Downloading dataset and checkpoint
To download dataset, you should run
```shell 
chmod +x scripts/download_dataset.sh
poetry run sh scripts/download_dataset.sh
```
Since this script is reordering and renaming files, installing dataset with other methods may not work.  
To download model chekpoint, run
```shell 
chmod +x scripts/download_best_model.sh
poetry run sh scripts/download_best_model.sh
```

## Best model
#### Description
The model is a implementation of RawNet2 

#### Training
To train this model independently, you should run
```shell 
poetry run python train.py
```
The config for this model is file /src/config.yaml 

#### Testing
To test specific audio file, you should change add it to the folder test_audio/ and run
```shell 
poetry run python test.py
```
Bonafide probability will be printed and will appear in result.json file 

## Credits
This homework was done by Ivan Dmitriev

This repository is based on a fork
of [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.
