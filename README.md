# DCPPS

DCPPS provides the prediction of kinase-specific phosphorylation sites using dynamic embedding and cross-representation interaction

# Requirement
```env
python == 3.7

keras == 2.4.0

tensorflow == 2.4.0

numpy >= 1.8.0

backend == tensorflow
```

# Predict For Your Test Data
## Download the model weights of all kinase-specific phosphorylation sites datasets
The model weights were saved in the cloudy space: https://drive.google.com/file/d/1ikwFfqmjzDwkxYBf57aUrP5C17bQbjZh/view?usp=sharing

## Predict
```shell
cd ./DCPPS

# If you want to predict kinase-specific sites, take MAPK as an example:
# First modify the kinase name in predict.py
# Then, run
python predict.py
```

# Train
```shell
python train.py
```
