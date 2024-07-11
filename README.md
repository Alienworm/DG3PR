# Code for the paper "DG3PR: Dynamic Global Graph Guided Price-aware Recommendation"
## Requirements
- python==3.7
- torch==1.11.0
- torch-geometric==2.3.0

## Data Preparation
Because the data is sensitive, we cannot provide the original data.
You should prepare the data in the following format:
```json
{
    "customer": "customer_column", 
    "breed": "item_column",
    "breed_class": "item_category_column",
    "guide_price": "item_price_column"
}
```

## Data Preprocessing
```shell
python raw_data_process.py
```

## Training
```shell
python train_dg3pr.py
```
