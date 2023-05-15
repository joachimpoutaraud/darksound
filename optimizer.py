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

import optuna
from optuna.trial import TrialState

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from networks.protonet import PrototypicalNetworks
from networks.matchnet import MatchingNetworks
from networks.relatnet import RelationNetworks

from trainer import trainer
from config import load_config
from dataset.sampler import TaskSampler
from dataset.darksound import Darksound

import warnings
warnings.filterwarnings("ignore")


def objective(trial):
    """Objective function to be optimized by Optuna.
    Hyperparameters chosen to be optimized: optimizer and learning rate.
    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """
    # Load the few-shot classifier
    if params['PARAMS_MODEL']['MODEL'] == 'matching':
        few_shot_classifier = MatchingNetworks(model, use_softmax=True).to(device)
    if params['PARAMS_MODEL']['MODEL'] == 'prototypical':
        few_shot_classifier = PrototypicalNetworks(model, use_softmax=True).to(device)
    if params['PARAMS_MODEL']['MODEL'] == 'relation':
        # Relation module takes feature maps as input
        relation_module = create_feature_extractor(model, return_nodes=['layer4.0.conv2'])
        few_shot_classifier = RelationNetworks(relation_module, use_softmax=False).to(device)
    
    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Load the optimizer, loss and scheduler
    if  params['PARAMS_MODEL']['TRAINING'] == 'classical':
        # Optimize the loss with the pre-trained ResNet18
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

    if  params['PARAMS_MODEL']['TRAINING'] == 'episodic':
        # Optimize the loss with the Few-Shot classifier
        optimizer = getattr(torch.optim, optimizer_name)(few_shot_classifier.parameters(), lr=lr)
        if params['PARAMS_MODEL']['MODEL'] == 'relation':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
            
    train_scheduler = ReduceLROnPlateau(optimizer, factor=params['PARAMS_MODEL']['SCHEDULER_FACTOR'], patience=params['PARAMS_MODEL']['SCHEDULER_PATIENCE'])

    # Training of the model
    for epoch in range(params['PARAMS_MODEL']['N_EPOCHS']):
        print(f"Epoch {epoch}")
        if params['PARAMS_MODEL']['TRAINING'] == 'classical':
            training_loss, training_accuracy = trainer(model, train_loader, optimizer, criterion, train=True, training=params['PARAMS_MODEL']['TRAINING'], device=device)
            model.fc = nn.Flatten() # remove fully connected layer for validating

            # Load the few-shot classifier
            if params['PARAMS_MODEL']['MODEL'] == 'matching':
                few_shot_classifier = MatchingNetworks(model, use_softmax=True).to(device)
            if params['PARAMS_MODEL']['MODEL'] == 'prototypical':
                few_shot_classifier = PrototypicalNetworks(model, use_softmax=True).to(device)
            if params['PARAMS_MODEL']['MODEL'] == 'relation':
                # We take only few convolutional layers of the ResNet as Relation module takes feature maps as input
                relation_model = create_feature_extractor(model, return_nodes=['layer4.0.conv2'])
                few_shot_classifier = RelationNetworks(relation_model, use_softmax=False).to(device)

            validation_loss, validation_accuracy = trainer(few_shot_classifier, val_loader, optimizer, criterion, train=False, device=device)
            model.fc = nn.Linear(embedding_dim, n_classes, device=device)

        if params['PARAMS_MODEL']['TRAINING'] == 'episodic':
            training_loss, training_accuracy = trainer(few_shot_classifier, train_loader, optimizer, criterion, train=True, training=params['PARAMS_MODEL']['TRAINING'], device=device)
            validation_loss, validation_accuracy = trainer(few_shot_classifier, val_loader, optimizer, criterion, train=False, device=device)

        # Decrease the learning rate
        train_scheduler.step(training_loss)
        # For pruning (stops trial early if not promising)
        trial.report(validation_accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return validation_accuracy


if __name__ == '__main__':

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Import config file and load parameters
    CONFIG_FILE = 'config.yaml'
    params = load_config(CONFIG_FILE)

    # --- PARAMETERS ----------------------------------------------------------
    number_of_trials = 50                # Number of Optuna trials
    # -------------------------------------------------------------------------

    # Make runs repeatable
    random_seed = params['RANDOM_SEED']
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    # CREATE THE TASK SAMPLER AND DATALOADERS                
    # -------------------------------------------------------------------------

    # Training 
    if params['PARAMS_MODEL']['TRAINING'] == 'classical':
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

    # Validation sampler
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

    # Create an Optuna study to maximize validation accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)

    # -------------------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------------------

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv(f"optimization-{params['PARAMS_MODEL']['TRAINING']}-{str(params['PARAMS_MODEL']['N_WAY'])}-way-{str(params['PARAMS_MODEL']['N_SHOT'])}-shot.csv", index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))
    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
