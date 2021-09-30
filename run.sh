export TRAINING_DATA=Categorical_Encoding/train_folds.csv
export TEST_DATA=input/cat-in-the-dat/test.csv
# export MODEL="extratrees"
export MODEL=$1

# FOLD=0 python -m Categorical_Encoding.src.train
# FOLD=1 python -m Categorical_Encoding.src.train
# FOLD=2 python -m Categorical_Encoding.src.train
# FOLD=3 python -m Categorical_Encoding.src.train
# FOLD=4 python -m Categorical_Encoding.src.train

python -m Categorical_Encoding.src.predict

# sh Categorical_Encoding/run.sh randomforest