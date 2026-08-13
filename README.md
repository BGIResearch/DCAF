# DCAF
**Data-Criterion-model associated Framework for full cycle of single-cell pertubation modeling!**

<img width="2680" height="3424" alt="fig1" src="https://github.com/user-attachments/assets/e545ef68-42b6-4214-9d9b-44fb659a553e" />



### Installation of DCAF-data and DCAF-criterion
```bash
git clone https://github.com/BGIResearch/DCAF.git

cd DCAF

conda create -n DCAF_data_cri python=3.9

pip install .

pip install -r requirements.txt
```

### Activate environment for DCAF-data and DCAF-criterion
```bash
conda activate DCAF_data_cri
```

### Demo for DCAF-data
```bash
cd DCAF

python run_recover.py -m demo
```
Dataset Tian, derived from [paper](https://pubmed.ncbi.nlm.nih.gov/34031600/) is applied for this demonstration. 

### Use DCAF-data to recover control and case status of your own data
```bash
cd DCAF

python run_recover.py -ctrl NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/data
                      -case NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/data
```
Control and case npz files shall be placed in the same path. They don't necessarily be  of the same size.

### Demo for DCAF-criterion
```bash
cd DCAF

python run_measure.py -m demo
```
Synthetic dataset using our article (a link to be inserted) is applied for this demonstration.

### Use DCAF-criterion to assess accuracy of your own output, using your own dataset
```bash
cd DCAF

python run_measure.py -ctrl NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/data
                      -case NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/data
                      -pred NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/data
```
Control, case, and predicted npz files shall be placed in the same path. They shall be of the same size.

### Installation of DCAF-model
```bash
git clone https://github.com/BGIResearch/DCAF.git
cd DCAF

conda create -n DCAF_model python=3.9
conda activate DCAF_model

pip install .
pip install -r requirements.txt (Need another requirement file)
```

### Activate environment for DCAF-model
```bash
conda activate DCAF_model
```

### Demo for DCAF-model inference
Download fintuned models and dictionary, from [drive](https://drive.google.com/drive/folders/1tuOBDReWq4vyj-B5G_w1EZg_PVpjBoAz?usp=sharing) to infer_by_ctrl/ours/data；

Use notebook for inference, at https://github.com/BGIResearch/DCAF/blob/main/infer_each_cell.ipynb
