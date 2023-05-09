# UNSUPEVISED META-EMBEDDING FOR BIRD SONGS CLUSTERING IN SOUNDSCAPE RECORDINGS

Amazonian forests are threatened by numerous anthropogenic pressures not visible by satellite imagery, such as over-hunting or undercover forest degradation. Knowledge of the effects of these degradations is essential for an effective local conservation policy. However, these effects can only be assessed using quantitative methods for monitoring biodiversity in the field. In recent years, ecoacoustics has offered an alternative to traditional techniques with the development of Passive Acoustic Monitoring (PAM) systems allowing, among other things, to automatically monitor species that are difficult to identify by observers, such as crepuscular and nocturnal tropical birds. Although the use of such systems makes it possible to acquire large sets of data collected in the field, it is often difficult to process these data because they generally represent several thousand hours of recordings that need to be annotated and validated manually by an expert with in-depth knowledge of the phenology and behavior of the species studied. This repository is based on a Master's thesis in order to develop a new method to facilitate the work of ecoacousticians in managing large unlabeled acoustic datasets and to improve the identification of potential new taxa. Based on the advancement of Meta-Learning methods and unsupervised learning techniques integrated in the Deep Learning (DL) framework, we propose the Meta Embedded Clustering (MEC) method to progressively discover and improve the inherent structure of unlabeled data. 

## Installation
Download [Anaconda](https://www.anaconda.com/products/distribution) and prepare your environment using the command line.
```
conda create --name darksound python=3.8
conda activate darksound
conda install -c conda-forge hdbscan # on Windows
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraires
```
pip install -r requirements.txt
```

## Usage
### Darksound Dataset
Darksound dataset is build as an open-source and code-based dataset for the evaluation of Meta-Learning algorithms in the context of ecoacoustics with DL. The particularity of this dataset is that it is composed of acoustic units, also called Regions of Interest (ROIs), of 290 nocturnal and crepuscular bird species living in tropical environments. All the ROIs in the Darksound dataset have a sampling rate of 48 kHz and are faded in and out to avoid aliasing effects due to window effects. Moreover, each ROI is padded to a maximum duration of 3 seconds in order to obtain input images of equal size for training the model. The dataset is easily accessible and downloadable on [Kaggle](https://www.kaggle.com/datasets/joachipo/darksound) or can be directly downloaded using the programming language Python:

```python
from dataset.darksound import Darksound
from torchvision.models import ResNet18_Weights
from torchvision import transforms

weights = ResNet18_Weights.IMAGENET1K_V1 # Load ResNet18 weights for transformation

train_set = Darksound(split='train', transform=transforms.Compose([weights.transforms()]), download=True)
tval_set = Darksound(split='val', transform=transforms.Compose([weights.transforms()]), download=True)
test_set = Darksound(split='test', transform=transforms.Compose([weights.transforms()]), download=True)
```

### Training Meta-Learning algorithms
It is possible to train metric-learning based algorithms ([Matching Networks](https://arxiv.org/pdf/1606.04080.pdf), [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf) or [Relation Networks](https://arxiv.org/pdf/1711.06025.pdf)) by indicating the desired parameters in the `config.yaml` file.   

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate