# DCAF
**Data-Criterion-model associated Framework for full cycle of single-cell pertubation modeling!**

<img width="2680" height="3424" alt="fig1" src="https://github.com/user-attachments/assets/e545ef68-42b6-4214-9d9b-44fb659a553e" />



### Installation
```bash
git clone https://github.com/BGIResearch/DCAF.git

cd DCAF

pip install .

pip install -r requirements.txt
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

python run_recover.py -ctrl NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/DATA
                      -case  NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/DATA
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

python run_measure.py -ctrl NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/DATA
                      -case NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/DATA
                      -pred NPZ_FULL_PATH_OR_ITS_FILE_NAME_UNDER_/DATA
```
Control, case, and predicted npz files shall be placed in the same path. They shall be of the same size.


### Demo for DCAF-model inference
Download fintuned models and dictionary, from [drive](https://drive.google.com/drive/folders/1tuOBDReWq4vyj-B5G_w1EZg_PVpjBoAz?usp=sharing) to infer_by_ctrl/ours/data；

Use notebook for inference, at https://github.com/BGIResearch/DCAF/blob/main/infer_each_cell.ipynb
