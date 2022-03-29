# Tumor classification

This project built for brain tumor detection and classification. We are trying to classify 3 types of tumor in human brain.

1. pituitary_tumor
2. glioma_tumor
3. meningioma_tumor

In this project we are using Resnet-50 model architecture. This project is implemented with MLflow.


# STEPS

## step 01 - Clone the new repository

## step 02 - Create a conda environment after opening the repository in VSCODE
```
conda create --prefix ./env python=3.7 -y
```
## activate environment
```
conda activate ./env
```
### or
```
source activate ./env
```

## STEP 03- install the requirements
```
pip install -r requirements.txt
```
## step 04- install pytorch 11.3

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
## or use init_setup.sh if not want run step 01 to step 05
### in bash terminal use below command
```
bash init_setup.sh
```
## step 05- install setup.py
```
pip install -e .
``` 