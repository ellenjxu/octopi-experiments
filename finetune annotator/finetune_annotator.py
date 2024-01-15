import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import time

from sklearn.model_selection import train_test_split
from cleanlab.multiannotator import get_label_quality_scores, get_active_learning_scores

import models
import utils
    
def active_query(model, images, batch_size=32):
    """
    Selects a subset of the o.o.d. data to interactively query user to obtain labels, based on active learning.
    """
    # generate predictions and features
    pred_probs_unlabeled, features = utils.generate_predictions_and_features(model,images,batch_size)
    pred_probs_unlabeled = pred_probs_unlabeled.squeeze()

    # compute active learning scores
    _, active_learning_scores_unlabeled = get_active_learning_scores(
        pred_probs_unlabeled=pred_probs_unlabeled
        # df_labeled['label'].to_numpy(), pred_probs_unlabeled=pred_probs_unlabeled  # TODO: in the future can also choose relabeling from labeled dataset (may need ood preds)
    )

    # active_learning_scores_unlabeled[:5]

    return np.argsort(active_learning_scores_unlabeled)[:batch_size]

def update(model, images, labels, num_epochs=10, learning_rate=0.1):   # TODO: simple update with higher LR, can weight based on scores/how off

    """
    Updates model with new data + labels from active query.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if images.dtype == np.uint8:
        images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

    dataset = TensorDataset(
        torch.from_numpy(images), 
        torch.from_numpy(labels)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        for inputs, labels in dataloader: 

            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad() 
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
    return model

#------------------------------------------------------------

dir_path = 'npy_v2/'

trained_on = ['PAT-070-3_2023-01-22_15-24-28.812821.npy',
                'PAT-071-3_2023-01-22_15-47-3.096602.npy',
                'PAT-072-1_2023-01-22_17-17-58.363496.npy',
                'PAT-073-1_2023-01-22_16-32-5.192404.npy',
                'PAT-074-1_2023-01-22_16-55-50.887780.npy',
                'PBC-404-1_2023-01-22_19-09-9.267139.npy',
                'PBC-502-1_2023-01-22_17-49-38.429975.npy',
                'PBC-800-1_2023-01-22_21-30-44.794123.npy',
                'PBC-801-1_2023-01-22_22-06-18.047215.npy',
                'PBC-1023-1_2023-01-22_19-59-54.633046.npy']

files_unlabelled = [f for f in os.listdir(dir_path) if f.endswith('.npy') and f not in trained_on and not f.startswith('SBC')]
# TODO: if desired, fold in known SBC negatives (right now just using positive patient samples to finetune on)

images_unlabelled = []
for f in files_unlabelled[:10]: # TODO: do it in batches, 41 slides total
    images_unlabelled.append(np.load(dir_path + f))

images_unlabelled = np.vstack(images_unlabelled)
print(images_unlabelled.shape)

# Train model on labeled data and get predicted class probabilites for unlabeled data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.ResNet(model='resnet34', n_channels=4, n_filters=64, n_classes=3, kernel_size=3, stride=1, padding=1)
model.load_state_dict(torch.load('tests/model_resnet34_1704164366.7755363.pt'))
model = model.to(device)

# run active learning rounds

# num_rounds = 10
# for _ in range(num_rounds):
    # get next batch to label
next_to_label = active_query(model, images_unlabelled, 32)
print(next_to_label)
images = images_unlabelled[next_to_label]

# get labels from user
preds, _ = utils.generate_predictions_and_features(model, images, 32) # TODO: already got this from active_query, can just pass in
labels = np.random.choice(2, len(images)) # TODO: get labels from user using interactive annotator

# update model
model = update(model, images, labels, 10, 0.1)

# remove from unlabelled
images_unlabelled = np.delete(images_unlabelled, next_to_label, axis=0)

    # TODO: evaluate on hold-out set, or stop when user no longer needs to correct

# np.save(model, 'model_active_learning.npy')