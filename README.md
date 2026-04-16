# DeepCDA Reproduction

Download the dataset from here:\
https://drive.google.com/open?id=15KotSJWknMOAnHM68RpOh_rqMISsMwsE
---
## Environment setup
Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) to create a `conda` environment.\
To create a conda environment and install libraries used in this project do the following(make sure you are inside the DeepCDA_Reproduction folder): 
```bash
conda create -n deepcda python=3.11
conda activate deepcda
pip install -r requirements.txt
```
**NOTE:** In `requirements.txt` file the version number for the libraries are taken from a macOS device. If using device running a different operating system, it is possible that version on Mac for a library is not available for that operating system. In this case, try removing the version numbers from libraries in the `requirements.txt` file.

---
## Encoder
### Training
To train the encoder update the `configs.yaml` folder to set the value of protein, smiles filter sizes, embedding dimension, channel dimension, number of training epoch, learning rate and other relevant hyperparameters. Remember to set `train` to `True`, and the hyperparmeters should be in list format where ever needed as shown in the example `configs.yaml` file. After saving the file run the following command: 
```python
python3 main.py --config_path configs.yaml
```
### Testing

To test the encoder set `test` to `True` in config file. Remember to use the hyperparameters that you used while training, else model weights will not match. After this run the same command: 
```python
python3 main.py --config_path configs.yaml
```
---
## Domain Adaptation

To run the domain adaptation codes first make sure that you have trained and saved the encoder weights. Next edit the `ada_config.yaml` folder and make sure the enter the correct folder path to where the model weights are stored. After saving the config file run the following command on terminal: 
```python
python3 ada_train.py --config_path ada_config.yaml
```

To test the domain adaptation results we will make use of the testing script written for the encoder training part. Make sure that `save_folder` points to folder where domain adaptation weights trained have been save. Set `test` to `True` and then execute the following command in terminal:
```python
python3 main.py --config_path configs.yaml
```

The metrics should be stored in a `metrics.csv` file made inside your current folder. 
