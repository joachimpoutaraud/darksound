"""
Copyright [2023] [Poutaraud]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Tuple

import hdbscan
from metrics import ACC, DBCV

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

import librosa
import torchaudio
import torchaudio.transforms as T
from audiomentations import Compose, Gain, AddGaussianNoise, AirAbsorption, TimeStretch, Trim

from trainer import trainer
from config import load_config
from dataset.darksound import Darksound
from dataset.sampler import TaskSampler

from networks.protonet import PrototypicalNetworks
from networks.matchnet import MatchingNetworks

import warnings
warnings.filterwarnings("ignore")


class PseudoDataset(VisionDataset):
    def __init__(self, data, pseudo_labels, augmentation):                
        # Remove noise and assign pseudo labels
        idx = np.where(pseudo_labels >= 0)[0]
        self.data = [(data[i][0], pseudo_labels[i]) for i in idx]
        
        if augmentation:
            files = [data._flat_species_files[i][0] for i in idx]
            cwd = os.getcwd()
            os.chdir(data.target_folder)
            
            print(f'Initial number of samples: {len(self.data)}')
            for i in tqdm(range(49), desc='Augmenting the pseudo labeled set'):
                for file in glob.glob("*/*/*.wav"):
                    try:
                        x = files.index(os.path.basename(file))
                        label = pseudo_labels[idx][x]
                        Y_aug = augmented(file, label, transform=transforms.Compose([weights.transforms()]))
                        self.data.append(Y_aug)
                    except ValueError:
                        pass
            os.chdir(cwd)
            print(f'Number of samples after augmentation: {len(self.data)}\n')
             
    def __getlabel__(self) -> list:
        return [instance[1] for instance in self.data]
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.data[index]
        return image, label
    def __len__(self) -> int:
        return len(self.data)
    
def augmented(file, label, transform, pad=True, seconds=3, n_fft=1024, hop_length=512, n_mels=128, margin=(1.0,5.0)):
    # Load audio file and normalize it using torch
    y, sr = torchaudio.load(file, normalize=True)
    # Fade in and out to avoid aliasing from window effects.
    fade_transform = T.Fade(fade_in_len=int(sr/10), fade_out_len=int(sr/10), fade_shape='half_sine')
    y = fade_transform(y)
         
    # Augment the samples using audiomentations
    augment = Compose([
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
                    AirAbsorption(),
                    # Speed up or slow down the sample by 25%
                    TimeStretch(min_rate=0.75, max_rate=1.25, p=1.0), 
                    Trim(top_db=30.0, p=1.0)
                    ])  
    
    y = torch.from_numpy(augment(samples=y.numpy(), sample_rate=sr))
    
    if pad:
        samples = sr*seconds
        if y.shape[1] >= (samples):
            y.resize_(1, samples)
        else:
            diff = (samples) - y.shape[1]
            pad = nn.ConstantPad1d((int(np.ceil(diff/2)), int(np.floor(diff/2))), 0)
            y = pad(y)

    spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        n_mels=n_mels)
    
    Y = torch.flip(spectrogram(y), dims=[0, 1]).numpy()[0]
    
    # Compute HP Source Separation
    H, P = librosa.decompose.hpss(Y, margin=margin)
    # Convert amplitude spectrogram to dB-scaled 
    Y = librosa.amplitude_to_db(Y, ref=np.max)
    H = librosa.amplitude_to_db(H, ref=np.max)
    P = librosa.amplitude_to_db(P, ref=np.max)

    # Compute delta features
    D = librosa.feature.delta(Y)
    
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    # Normalize spectrogram
    H = scale_minmax(H)
    P = scale_minmax(P)
    D = scale_minmax(D)

    HPD = np.nan_to_num(np.transpose(np.asarray(np.dstack((H,P,D))), (2,0,1)))
    HPD = torch.from_numpy(HPD)
       
    return (transform(HPD), label)
    
def get_features(model, spectrogram, device, params):
    # Extract the features from the model
    features = model.backbone.forward(spectrogram.to(device).unsqueeze(dim=0)).squeeze(dim=0) 
#         features = model.encode_support_embeddings(model.backbone.forward(spectrogram.to(device).unsqueeze(dim=0))).squeeze(dim=0)
    # Detach and convert to numpy array 
    return features.detach().cpu().numpy()

def evaluation(features, test_set, cluster_labels):
    # Get the ground truth labels and predicted indexes 
    labels_true = np.array([x[1] for x in test_set])
    idx = np.where(cluster_labels >= 0)[0]
    
    y_true = labels_true[idx]
    y_pred = cluster_labels[idx]
    
    accuracy = ACC(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    dbcv = DBCV(features, cluster_labels) 
    return accuracy, nmi, ari, dbcv

def training(model, dataset, labels, params, device):
    # Create the pseudo-labeled dataset
    pseudo_set = PseudoDataset(dataset, labels, params['PARAMS_MODEL']['AUGMENTATION']) 
    
    pseudo_sampler = TaskSampler(pseudo_set, 
                                 n_way=len(np.unique(labels))-1, # Remove outliers
                                 n_shot=params['PARAMS_MODEL']['N_SHOT'], 
                                 n_query=params['PARAMS_MODEL']['N_QUERY'], 
                                 n_tasks=params['PARAMS_MODEL']['N_TASKS'])

    pseudo_loader = DataLoader(pseudo_set,
                               batch_sampler=pseudo_sampler,
                               num_workers=params['PARAMS_MODEL']['N_WORKERS'],
                               pin_memory=True,
                               collate_fn=pseudo_sampler.episode)
    
    # Load the clustering loss and the optimizer
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = Adam(model.parameters(), lr=params['PARAMS_MODEL']['LEARNING_RATE']) 
    
    for epoch in range(params['PARAMS_MODEL']['N_EPOCHS']):        
        print(f"Epoch {epoch}")  
        loss, accuracy = trainer(model, 
                                 pseudo_loader, 
                                 optimizer, 
                                 criterion, 
                                 train=True, 
                                 training=params['PARAMS_MODEL']['TRAINING'], 
                                 device=device)
            
    return model


if __name__ == '__main__':
    # Import config file and load parameters
    CONFIG_FILE = 'config.yaml'
    params = load_config(CONFIG_FILE)

    # Set the seed for all random packages that could possibly be used
    random_seed = params['RANDOM_SEED']
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the weights of the ResNet18
    weights = ResNet18_Weights.IMAGENET1K_V1

    # -------------------------------------------------------------------------
    # LOAD THE DATASET                
    # -------------------------------------------------------------------------

    test_set = Darksound(split='test', 
                        hpss=params['PARAMS_MODEL']['HPSS'], 
                        download=True, 
                        remove_background=params['PARAMS_MODEL']['REMOVE_BG'], 
                        transform=transforms.Compose([weights.transforms()]))
    
    # -------------------------------------------------------------------------
    # LOAD THE MODEL                
    # -------------------------------------------------------------------------

    # Load the ResNet18 model
    if params['PARAMS_MODEL']['PRETRAINED']:
        model = resnet18(weights=weights).to(device)
    if params['PARAMS_MODEL']['PRETRAINED'] == False:
        model = resnet18(weights=None).to(device)

    model.fc = nn.Flatten()
    # model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Flatten())
    
    # Load the Meta-learner
    if params['PARAMS_MODEL']['MODEL'] == 'matching':
        model = MatchingNetworks(model, use_softmax=True).to(device)
    if params['PARAMS_MODEL']['MODEL'] == 'prototypical':
        model = PrototypicalNetworks(model, use_softmax=True).to(device)
            
    # Load pre-trained model for fune-tuning on pseudo labeled set
    model.load_state_dict(torch.load(params['PARAMS_MODEL']['FEATURE_EXTRACTOR'], map_location=device)) 
    # Freeze layers of the backbone to reduce overfitting
#     for param in model.backbone.layer1.parameters(): 
#         param.requires_grad = False
#     for param in model.backbone.layer2.parameters():
#         param.requires_grad = False
#     for param in model.backbone.layer3.parameters():
#         param.requires_grad = False
#     for param in model.backbone.layer4.parameters():
#         param.requires_grad = False
        
    # -------------------------------------------------------------------------
    # ITERATIVE DEEP CLUSTERING               
    # -------------------------------------------------------------------------

    df = pd.DataFrame({'Clusters': pd.Series(dtype='int'),
                       'Accuracy': pd.Series(dtype='float'),
                       'NMI': pd.Series(dtype='float'),
                       'ARI': pd.Series(dtype='float'),
                       'DBCV': pd.Series(dtype='float'),
                      })
    metrics = []
    
    for n in range(params['PARAMS_MODEL']['N_ITERATIONS']):
        print(f'\n***** Iteration {str(n)} *****')
        
        # Extract the features from the model
        features = []
        for i in tqdm(range(len(test_set)), desc='Extracting features'):
            # Extracting features from the model
            features.append(get_features(model, test_set[i][0], device, params))

        print('Evaluating clustering performances:\n')
        # Normalize and cluster the data
        features = MinMaxScaler().fit_transform(np.array(features))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(params['PARAMS_MODEL']['N_SHOT']+params['PARAMS_MODEL']['N_QUERY']), 
                                    prediction_data=True, 
                                    ).fit(features) 

        labels = clusterer.labels_
        
        # Evaluate the clustering performances
        accuracy, nmi, ari, dbcv = evaluation(features, test_set, labels)
        n_clusters = len(np.unique(labels))-1 # Remove outliers
        print(f'Total number of clusters: {n_clusters}')
        print(f'Accuracy: {accuracy}\nNMI: {nmi}\nARI: {ari}\nDBCV: {dbcv}\n')

        # Append performance metrics to dataframe
        performance_metrics = [n_clusters, accuracy, nmi, ari, dbcv]
        metrics.append(pd.DataFrame([performance_metrics], columns=["Clusters","Accuracy","NMI","ARI","DBCV"]))
        
        # Fine-tune the model on pseudo-labeled dataset
        model = training(model, test_set, labels, params, device)
        torch.save(model.state_dict(), f"models/idc-iteration-{str(n)}.pt") 
        torch.cuda.empty_cache()

    # Save the csv with clustering performances 
    df = pd.concat(metrics)
    df.to_csv('models/measurements/clustering_metrics.csv', index=False)
