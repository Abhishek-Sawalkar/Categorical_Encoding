export TRAINING_DATA=Categorical_Encoding/train_folds.csv
# export MODEL="extratrees"
export MODEL=$1

FOLD=0 python -m Categorical_Encoding.src.train


# sh Categorical_Encoding/run.sh randomforest