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

import random
import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from trainer import trainer
from config import load_config
from dataset.darksound import Darksound
from dataset.sampler import TaskSampler

from networks.protonet import PrototypicalNetworks
from networks.matchnet import MatchingNetworks
from networks.relatnet import RelationNetworks

from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

import warnings
warnings.filterwarnings("ignore")


def dataloader(train_set, val_set, params): 
    if params['PARAMS_MODEL']['TRAINING'] == 'classical':
        # Define the batch size according to the few-shot classification task
        batch_size = params['PARAMS_MODEL']['N_WAY'] * (params['PARAMS_MODEL']['N_SHOT'] + params['PARAMS_MODEL']['N_QUERY'])

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=params['PARAMS_MODEL']['N_WORKERS'],
            pin_memory=True,
            shuffle=True,
        )
    if params['PARAMS_MODEL']['TRAINING'] == 'episodic':
        train_sampler = TaskSampler(train_set, n_way=params['PARAMS_MODEL']['N_WAY'], 
                            n_shot=params['PARAMS_MODEL']['N_SHOT'], 
                            n_query=params['PARAMS_MODEL']['N_QUERY'], 
                            n_tasks=params['PARAMS_MODEL']['N_TASKS'])
        
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            num_workers=params['PARAMS_MODEL']['N_WORKERS'],
            pin_memory=True,
            collate_fn=train_sampler.episode,
        )

    # Validation samplers
    val_sampler = TaskSampler(val_set, n_way=params['PARAMS_MODEL']['N_WAY'], 
                            n_shot=params['PARAMS_MODEL']['N_SHOT'], 
                            n_query=params['PARAMS_MODEL']['N_QUERY'], 
                            n_tasks=params['PARAMS_MODEL']['N_TASKS'])

    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=params['PARAMS_MODEL']['N_WORKERS'],
        pin_memory=True,
        collate_fn=val_sampler.episode,
    )

    return train_loader, val_loader

# Write the recorded energy consumption to csv
csv_handler = CSVHandler('models/measurements/energy_consumption.csv')

@measure_energy(handler=csv_handler)
def training(model, few_shot_classifier, train_loader, val_loader, params, device):
    
    validation_loss, validation_accuracy = 0, 0

    if params['PARAMS_MODEL']['TRAINING'] == 'classical':
        training_loss, training_accuracy = trainer(model, train_loader, optimizer, criterion, train=True, training=params['PARAMS_MODEL']['TRAINING'], device=device)
        model.fc = nn.Flatten() # remove fully connected layer for validating with the few-shot classifier

        if params['PARAMS_MODEL']['VALIDATION']:
            # Load the few-shot classifier
            if params['PARAMS_MODEL']['MODEL'] == 'matching':
                few_shot_classifier = MatchingNetworks(model, use_softmax=True).to(device)
            if params['PARAMS_MODEL']['MODEL'] == 'prototypical':
                few_shot_classifier = PrototypicalNetworks(model, use_softmax=True).to(device)
            if params['PARAMS_MODEL']['MODEL'] == 'relation':
                # Relation module takes feature maps as input
                relation_module = create_feature_extractor(model, return_nodes=['layer4.0.conv2'])
                few_shot_classifier = RelationNetworks(relation_module, use_softmax=False).to(device)

            validation_loss, validation_accuracy = trainer(few_shot_classifier, val_loader, optimizer, criterion, train=False, device=device)
            model.fc = nn.Linear(embedding_dim, n_classes, device=device)

    if params['PARAMS_MODEL']['TRAINING'] == 'episodic':
        training_loss, training_accuracy = trainer(few_shot_classifier, train_loader, optimizer, criterion, train=True, training=params['PARAMS_MODEL']['TRAINING'], device=device)
        if params['PARAMS_MODEL']['VALIDATION']:
            validation_loss, validation_accuracy = trainer(few_shot_classifier, val_loader, optimizer, criterion, train=False, device=device)

    return model, few_shot_classifier, training_loss, training_accuracy, validation_loss, validation_accuracy 


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

    train_set = Darksound(split='train', 
                        hpss=params['PARAMS_MODEL']['HPSS'], 
                        download=True, 
                        remove_background=params['PARAMS_MODEL']['REMOVE_BG'],
                        transform=transforms.Compose([weights.transforms()]))
    val_set = Darksound(split='val', 
                        hpss=params['PARAMS_MODEL']['HPSS'], 
                        download=True, 
                        remove_background=params['PARAMS_MODEL']['REMOVE_BG'], 
                        transform=transforms.Compose([weights.transforms()]))

    # -------------------------------------------------------------------------
    # TRAIN AND VALIDATE THE MODEL                
    # -------------------------------------------------------------------------

    # Load the ResNet18 model
    if params['PARAMS_MODEL']['PRETRAINED']:
        model = resnet18(weights=weights).to(device)
    if params['PARAMS_MODEL']['PRETRAINED'] == False:
        model = resnet18(weights=None).to(device)

    if params['PARAMS_MODEL']['MODEL'] == 'relation':
        # Freeze the layers of the ResNet18
        for param in model.parameters():
            param.requires_grad = False

    if params['PARAMS_MODEL']['TRAINING'] == 'classical':
        embedding_dim = model.fc.in_features
        n_classes = len(set(train_set.__getlabel__()))
        model.fc = nn.Linear(embedding_dim, n_classes)
    if params['PARAMS_MODEL']['TRAINING'] == 'episodic':
        model.fc = nn.Flatten()

    if params['PARAMS_MODEL']['TENSORBOARD']:
        # Load the summary writer
        tb_writer = SummaryWriter()
        
    best_validation_accuracy = 0.0
    best_epoch = -1

    # Load the few-shot classifier
    if params['PARAMS_MODEL']['MODEL'] == 'matching':
        few_shot_classifier = MatchingNetworks(model, use_softmax=True).to(device)
    if params['PARAMS_MODEL']['MODEL'] == 'prototypical':
        few_shot_classifier = PrototypicalNetworks(model, use_softmax=True).to(device)
    if params['PARAMS_MODEL']['MODEL'] == 'relation':
        # Relation module takes feature maps as input
        relation_module = create_feature_extractor(model, return_nodes=['layer4.0.conv2'])
        few_shot_classifier = RelationNetworks(relation_module, use_softmax=False).to(device)

    # Load the optimizer, loss and scheduler
    if  params['PARAMS_MODEL']['TRAINING'] == 'classical':
        # Optimize the loss with the pre-trained ResNet18
        optimizer = Adam(model.parameters(), lr=params['PARAMS_MODEL']['LEARNING_RATE'])
        best_state = model.state_dict()
        criterion = nn.CrossEntropyLoss()

    if  params['PARAMS_MODEL']['TRAINING'] == 'episodic':
        # Optimize the loss with the Few-Shot classifier
        optimizer = Adam(few_shot_classifier.parameters(), lr=params['PARAMS_MODEL']['LEARNING_RATE'])
        best_state = few_shot_classifier.state_dict()
        if params['PARAMS_MODEL']['MODEL'] == 'relation':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

    train_scheduler = ReduceLROnPlateau(optimizer, factor=params['PARAMS_MODEL']['SCHEDULER_FACTOR'], patience=params['PARAMS_MODEL']['SCHEDULER_PATIENCE'])

    for epoch in range(params['PARAMS_MODEL']['N_EPOCHS']):
        print(f"Epoch {epoch}")

        # Load different train and validation sampler for each epoch
        train_loader, val_loader = dataloader(train_set, val_set, params)
        model, few_shot_classifier, training_loss, training_accuracy, validation_loss, validation_accuracy = training(model, few_shot_classifier, train_loader, val_loader, params, device)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            if params['PARAMS_MODEL']['TRAINING'] == 'classical':
                best_state = model.state_dict()
            if params['PARAMS_MODEL']['TRAINING'] == 'episodic':
                best_state = few_shot_classifier.state_dict()
            best_epoch = epoch

        # Early stopping
        elif epoch - best_epoch > params['PARAMS_MODEL']['EARLY_STOP_THRESH']:
            print("Early stopped training at epoch %d" % epoch)
            break  # terminate the training loop

        if params['PARAMS_MODEL']['TENSORBOARD']:
            # Add scalar to the tensorboard writer
            tb_writer.add_scalar("Loss/Train", training_loss, epoch)
            tb_writer.add_scalar("Accuracy/Train", training_accuracy, epoch)
            tb_writer.add_scalar("Loss/Val", validation_loss, epoch)
            tb_writer.add_scalar("Accuracy/Val", validation_accuracy, epoch)
            # Decrease the learning rate
            train_scheduler.step(training_loss)
    
    if params['PARAMS_MODEL']['ENERGY_CONSUMPTION']:
        # Write the recorded energy consumption to csv
        csv_handler.save_data()

    # Load the best model's state and save it
    if params['PARAMS_MODEL']['TRAINING'] == 'classical':
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), 
                f"models/{params['PARAMS_MODEL']['MODEL']}-networks-{str(params['PARAMS_MODEL']['N_WAY'])}way-{str(params['PARAMS_MODEL']['N_SHOT'])}shot-{params['PARAMS_MODEL']['TRAINING']}.pt")
    if params['PARAMS_MODEL']['TRAINING'] == 'episodic':
        few_shot_classifier.load_state_dict(best_state)
        torch.save(few_shot_classifier.state_dict(), 
                f"models/{params['PARAMS_MODEL']['MODEL']}-networks-{str(params['PARAMS_MODEL']['N_WAY'])}way-{str(params['PARAMS_MODEL']['N_SHOT'])}shot-{params['PARAMS_MODEL']['TRAINING']}.pt")
