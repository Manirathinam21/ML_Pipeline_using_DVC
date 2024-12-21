# ML_Pipeline_using_DVC

## How to Run?

conda create -n test python=3.11 -y

conda activate test

pip install -r requirements.txt

## DVC commands
- git init
- dvc init
- dvc repro         ( To run the ML pipeline automatically)
- dvc dag           ( To see the ML pipeline diagram)
- dvc metrics show  (To see the Model evaluation metrics) 