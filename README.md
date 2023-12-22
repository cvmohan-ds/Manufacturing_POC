# Manufacturing POC

Manufacturing POC is an ML implementation for customer order amount prediction using ExtraTreeRegressor.

## Data
Please get the data for this project Oringinal Datset which was used in EDA [here](https://drive.google.com/file/d/14_n9Hdxzbf6utHGey5LtsOjb3gaFXCH4/view?usp=sharing) and preprocessed data for ML pipeline [here](https://drive.google.com/file/d/1LTPFptRU82rCB9j802anwPdUSTFWIVGG/view?usp=sharing).

## Get it working on your Local

- Create a virtual environment

```bash
python3 -m venv venv
```
- Clone the project
```bash
git clone https://github.com/cvmohan-ds/Manufacturing_POC.git .
```
- Install the reqirements.txt
```bash
pip install -r cust_amt_pred_prod_sample/requirements.txt
```
- put the downloaded preprocessed data into data folder under cust_amt_pred_prod_sample directory
```bash
cd cust_amt_pred_prod_sample 
mkdir data
cp <your path>/preprocessed_data.csv ./data
```
- Run the cust_amt_pred_prod_sample/notebooks/jupyter.ipynb notebook interactively. (I use vscode as my IDE)

## Once you have run your Notebook
- View the experiment in MLFlow UI (You should be in cust_amt_pred_prod_sample directory)

```bash
mlflow ui  --backend-store-uri sqlite:///metadata/mlflow/mlruns.db  --default-artifact-root ./metadata/mlflow/mlartifacts  --host localhost
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
