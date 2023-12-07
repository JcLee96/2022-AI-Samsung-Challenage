# 2022-AI-Samsung-Challenage
<div align="center">

# 2022-AI-Samsung-Challenage(DACON): 3D Metrology

</div>

## Datasets

All the models in this project were evaluated on the following datasets:

- [DACON](https://dacon.io/en/competitions/official/235954/data) (DACON Site)


## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/Leejucheon96/2022-AI-Samsung-Challenage.git


# [OPTIONAL] create conda environment
conda create -n 2022samsung python=3.8
conda activate 2022samsung
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -r requirements.txt

```

## Repository
```
/config: data and model parameter setting
/scripts: .sh file
/src: data load and augmentation, model code
```
 
## How to run
```
sh script.sh
```


