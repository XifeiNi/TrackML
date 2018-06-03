# Particle Identification Challenge

Assignment 2, COMP9417: Kaggle competition (https://www.kaggle.com/c/trackml-particle-identification). The challenge is to build an algorithm that quickly reconstructs particle tracks from 3D points left in the silicon detectors. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes.

### Prerequisites

- Python 3.0+ either locally or in a virtual environment.
- Datasets from kaggle: https://www.kaggle.com/c/trackml-particle-identification/data
- Packages required:
    - `trackml`: https://github.com/LAL/trackml-library.git This is the competition-specific package. 
    - `numpy`, `pandas`, `tqdm`, `scikit_learn` for our current best-performing solution `train_trackml-DBSCAN.py` (see the next section for how to install).
    - The other version, `train_trackml-HDBSCAN.py`, which gives a lower score than the DBSCAN version, requires an extra package `hdbscan`. However this package is **not able to be installed on CSE lab computers unless with sudo permission**. 

### Installing

First, clone this repository to your local machine. You may skip this step if you already have all the files: two python scripts and a `requirements.txt`.

```
git clone https://github.com/XifeiNi/TrackML.git
```

There is a `requirements.txt` bundled in this repository that lists the required packages you need to install. These are all compatible with the CSE lab environment. They can be installed using the following commands:

```
pip3 install --user -r requirements.txt
```

If that fails, you may use the following commands:

```
pip3 install --user git+https://github.com/LAL/trackml-library.git
pip3 install --user numpy pandas scikit_learn tqdm
```

Additionally, if you wish to run the alternative solution `train_trackml-HDBSCAN.py`, you will need to install one more package using the following command: 

```
pip3 install --user hdbscan
```
However note that **this package is not able to be installed on CSE lab computer unless with sudo permission**, because it requires certain linux packages that are missing on the lab computers. As it is not required in our current best-performing solution, we have not included it in `requirements.txt`.


## Running 
Run either script as usual:
```
python3 train_trackml-DBSCAN.py
```
```
python3 train_trackml-HDBSCAN.py
```


## Authors

* **Cecilia Ni**
* **Shahedul Islam** 
* **Kavi Shah**
* **Yi Xiao** 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Acknowledgments

* The kaggle community
* CERN: the European Organization for Nuclear Research

