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
cd DCAF/example

python run_recover.py
```
Dataset Tian, derived from [paper](https://pubmed.ncbi.nlm.nih.gov/34031600/) is applied for this demonstration. 

Susbtitute data/ctrl_tian.npz and data/case_tian.npz with your own control and case expression in .npz for usage.


### Demo for DCAF-criterion
```bash
cd DCAF/example

python run_measure.py
```
Synthetic dataset by data/simulate_data.py is applied for this demonstration.

### Demo for DCAF-model inference
Download fintuned models and dictionary, from [drive](https://drive.google.com/drive/folders/1tuOBDReWq4vyj-B5G_w1EZg_PVpjBoAz?usp=sharing) to infer_by_ctrl/ours/data；

Use notebook for inference, at https://github.com/BGIResearch/DCAF/blob/main/infer_each_cell.ipynb
