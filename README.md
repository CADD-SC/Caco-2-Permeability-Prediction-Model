# Caco-2-Permeability-Prediction-Model
Machine learning-based prediction model for Caco-2 Permeability prediction of drug candidates

## Introduction: ## 

Welcome to our repository, here we provide machine learning model to efficiently predict the Caco-2 permeability of target drug compounds in early stage drug discovery process. Caco-2 permeability refers to the ability of a drug compound to pass through Caco-2 cells, which are human epithelial colorectal adenocarcinoma cells. It provides essential insights into the oral absorption potential of drug candidates, helping to predict bioavailability, and optimize formulations.

## Classification criteria ##
The model uses an apparent permeability coefficient (Papp) threshold:

</strong> If <em>Papp</em> &ge; 8 &times; 10<sup>-6</sup> cm/s, the compound is <strong>Permeable</strong> and belongs to class 1. If <em>Papp</em> &lt; 8 &times; 10<sup>-6</sup> cm/s, it is <strong>Not Permeable</strong> and belongs to class 0.

## Dependencies ##

- Python ≥ 3.9
- scikit-learn ≥ 1.26.4
- numpy == 11.5.0
- hpsklearn == 1.0.3
- hyperopt == 0.2.7
- xgboost == 2.0.3
- rdkit
- pandas

## Execution ##
**To run the prediction:**

```
$ python model.py --prediction --file_name [filename] --model_path caco.pkl
```
<strong>Note:</strong> For the prediction step, prepare a .csv file containing SMILES without bioclass (e.g., test_set.csv)

**To run the validation:**

```
$ python model.py --validation --file_name [filename] --model_path caco.pkl
```
<strong>Note:</strong> For the validation step, prepare a .csv file containing SMILES with bioclass (0 or 1) (e.g., valid_set.csv)

**Output:**

Our model generates output in binary value (1 or 0), where 1 indicates compound to be permeable, while 0 indicates non-permeable

**Please ensure that all the necessary files (caco.pkl, data_preprocessing.py, scaler, features.txt, input_file.csv, model.py) are kept in the working directory**
