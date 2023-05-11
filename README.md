# UNSUPERVISED META-EMBEDDING FOR BIRD SONGS CLUSTERING IN SOUNDSCAPE RECORDINGS

This repository is associated to a Master's thesis carried out in 2023 at the University of Oslo (UiO) as part of the Music, Communication and Technology Master programme. The objective is to develop a new method to facilitate the work of ecoacousticians in managing large unlabeled acoustic datasets and to improve the identification of potential new taxa. Based on the advancement of Meta-Learning methods and unsupervised learning techniques integrated in the Deep Learning (DL) framework, the Meta Embedded Clustering (MEC) method is proposed in order to progressively discover and improve the inherent structure of unlabeled data. 

|![Meta Embedded Clustering (MEC)](https://raw.githubusercontent.com/joachimpoutaraud/darksound/master/notebooks/mec.jpg)|
|:--:| 
| Meta Embedded Clustering (MEC) method. (1) Data is passed through the initialized model. (2) Initial estimate of the non-linear mappings are computed to avoid the curse of dimensionality. (3) Clustering algorithm is performed on the latent space. (4) Pseudo-labeled dataset is built. (5) Model is fine-tuned on the pseudo-labeled dataset for *n* episodic tasks. |

## Installation
Download [Anaconda](https://www.anaconda.com/products/distribution) and prepare your environment using the command line.
```
conda create --name darksound python=3.8
conda activate darksound
```
If you are on Windows, it might be preferable to install the [hdbscan](https://hdbscan.readthedocs.io/en/latest/index.html) library beforehand.
```
conda install -c conda-forge hdbscan
```

Then, use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraires
```
pip install -r requirements.txt
```

## Usage
### Darksound Dataset
The Darksound Dataset is build as an open-source and code-based dataset for the evaluation of Meta-Learning algorithms in the context of ecoacoustics with DL. The particularity of this dataset is that it is composed of acoustic units, also called Regions of Interest (ROIs), of 290 nocturnal and crepuscular bird species living in tropical environments. All the ROIs in the Darksound dataset have a sampling rate of 48 kHz and are faded in and out to avoid aliasing effects due to window effects. Moreover, each ROI is padded to a maximum duration of 3 seconds in order to obtain input images of equal size for training the model. 

The dataset is easily accessible and downloadable on [Kaggle](https://www.kaggle.com/datasets/joachipo/darksound) or can be directly downloaded using the programming language Python:

```python
from dataset.darksound import Darksound
from torchvision.models import ResNet18_Weights
from torchvision import transforms

weights = ResNet18_Weights.IMAGENET1K_V1 # Load ResNet18 weights for transformation

train_set = Darksound(split='train', transform=transforms.Compose([weights.transforms()]), download=True)
val_set = Darksound(split='val', transform=transforms.Compose([weights.transforms()]), download=True)
test_set = Darksound(split='test', transform=transforms.Compose([weights.transforms()]), download=True)
```

### Meta-Learning algorithms
Meta-Learning algorithms are generally labeled as either metric-learning based or gradient-based meta-learner. In this repository, specific emphasis is placed on metric-learning based algorithms that are used for performing the experiments. More precisely, three different Meta-Learning algorithms ([Matching Networks](https://arxiv.org/pdf/1606.04080.pdf), [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf) or [Relation Networks](https://arxiv.org/pdf/1711.06025.pdf)) can be fine-tuned by indicating the desired parameters in the `config.yaml` file and running the following command:

```
python training.py
```
Evaluation of the classification performances of the Meta-Learning algorithms can be performed using this [notebook](https://github.com/joachimpoutaraud/darksound/blob/master/notebooks/02-model_evaluation.ipynb) and entering the path of the model weights available in the `models` folder.

### Meta Embedded Clustering (MEC)
Meta Embedded Clustering (MEC) method is proposed as an alternative to the DEC method introduced in ([Xie, et. al, 2016](https://arxiv.org/pdf/1511.06335.pdf)). MEC method is performed on the Darksound data set in order to refine the clusters of the 21 target species that are present in the test set. The objective of this method is to determine the final number of clusters in an unsupervised way in order to facilitate the identification and visualization of rare tropical bird species in unlabeled datasets. MEC method can be performed by indicating the desired parameters in the MEC section of the `config.yaml` file and running the following command:

```
python iterator.py
```
Evaluation of the clustering performances of the MEC method can be performed using this [notebook](https://github.com/joachimpoutaraud/darksound/blob/master/notebooks/03-clustering_evaluation.ipynb) and entering the path of the model weigths available in the `models` folder.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## Acknowledgements
We would like to thank authors from [EasyFSL](https://github.com/sicara/easy-few-shot-learning) for open-sourcing their code and publicly releasing checkpoints, and contributors to [Bambird](https://github.com/ear-team/bambird) for their excellent work in creating labelling function to build cleaner bird song recording dataset.
