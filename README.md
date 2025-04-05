# Dataset of Interest

The SARS-CoV-2 3CL protease (3CLPro) is a critical drug target for combating COVID-19 and future coronaviruses. As the enzyme responsible for viral replication, inhibiting 3CLPro disrupts the virus’s ability to multiply, making it a cornerstone for antiviral therapies.

I selected the SARSCoV2_3CLPro_Diamond dataset beacause despite  progress in vaccine development, effective antiviral drugs remains urgently needed. By focusing on this dataset, I aim to contribute to the discovery of novel inhibitors that could save lives and reduce healthcare burdens worldwide.

##  How to setup

Clone the repository using `git`

```
git clone https://github.com/olamidepeterojo/outreachy-contributions
```

Go to the root repository
```
cd outreachy-contributions
```

It's recommended developers make a virtual environment to install all required dependencies.

To create a virtual environment in the root repository, run the command below:

```
conda create --name myenv python=3.12
```

Activate your virtual environment in order for python to use it to manage dependencies.

```
conda activate myenv
```

Or, check the table titled "Command to activate virtual environment" [here](https://docs.python.org/3/library/venv.html#how-venvs-work) to find what works for your shell.

Forking my repository will give you acces to the TDC python package, but to know how to install it run:

```
pip install PyTDC
```

The `get_tdc_data.py` file contains code to obtain the SARSCoV2_3CLPro_Diamond dataset from HTS task in the single-instance prediction. if you desire to obtain another dataset task, still under single-instance prediction problem, you would have to change the import to the task you desire and the data.

## How to run

### Get Data Set

```
chmod +x /scripts/generate_data.sh
./scripts/generate_data.sh
```

## Dataset Analysis: SARS-CoV-2 3CLPro_Diamond (High-Throughput Screening)  

---

### **1. Verification as a Classification Problem**  
This dataset is a **binary classification task**, where the goal is to predict whether a compound is **active** (`Y=1`) or **inactive** (`Y=0`) against the SARS-CoV-2 3CL protease.  

**Key Evidence**:  
- **Target Variable**: Binary labels (`0` or `1`) explicitly provided for each compound.  
- **Class Distribution**:  
  - **Active (Y=1)**: 77 compounds (9% of the dataset).  
  - **Inactive (Y=0)**: 802 compounds (91% of the dataset).  
- **Suitability**: Classification aligns with the goal of identifying inhibitors (active compounds) from a chemical library.  
- **Evaluation Metrics**: Use **AUC-ROC**, **F1-score**, and **precision-recall curves** to handle class imbalance.  

---

### **2. Dataset Background and Endpoint**  
#### **Biological Context**  
- **Target**: SARS-CoV-2 3CL Protease (3CLPro),** a key enzyme for viral replication. Inhibiting this enzyme blocks viral maturation, making it a prime therapeutic target for COVID-19.  
- **Source**: Generated via **high-throughput screening (HTS)** at the [Diamond Light Source](https://www.diamond.ac.uk/), a synchrotron facility enabling rapid biochemical assays.  

#### **Endpoint Definition**  
- **Activity (Y=1)**: Compounds showing significant inhibition of 3CLPro in biochemical assays (e.g., IC50 ≤ 10 µM).  
- **Inactivity (Y=0)**: Compounds with no measurable inhibition (IC50 > 10 µM or no dose-response).  

#### **Dataset Curation**  
- **Size**: 879 compounds.  
- **Features**:  
  - `Drug_ID`: Unique compound identifier.  
  - `Drug`: SMILES strings encoding molecular structures.  
  - `Y`: Binary activity labels.  
- **Use Case**: Train models to prioritize novel 3CLPro inhibitors for experimental validation.  

---

### **3. Computational Feasibility Assessment**  
#### **Data Size and Complexity**  
- **Manageable Scale**: ~1k samples is tractable for most ML algorithms (e.g., Random Forests, SVMs) and deep learning on modest hardware.  
- **Featurization Workflow**:  
  - **Fingerprints**: Convert SMILES to Morgan fingerprints (e.g., 2048 bits) for traditional ML.  
  - **Graph-Based Models**: Libraries like `PyTorch Geometric` can be used to represent molecules as graphs for GNNs.  
  - **Descriptors**: Compute physicochemical properties (e.g., logP, molecular weight) via `RDKit`.  

---
  
This dataset is **viable for classification tasks** and computationally feasible for both traditional ML and deep learning. While class imbalance poses a challenge, it reflects real-world drug discovery constraints, making success here impactful for practical applications.  

---
