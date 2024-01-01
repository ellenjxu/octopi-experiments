"""Run 5 fold CV"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim import Adam
import copy
import time
import torch
import torch.nn as nn
import models
import utils
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def train_cv(model, images, labels_df, out_model_path, n_folds=5, batch_size=32, n_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # all_predictions = []
    # all_features = []
    # all_labels = []
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold = 0

    for train_index, test_index in kf.split(images):
        # print(train_index, test_index)
        fold += 1
        print(f"fold {fold}")

        # Use the indices to fetch the corresponding images and labels
        X_train, X_test = images[train_index], images[test_index]
        y_train, y_test = labels_df.loc[train_index, 'annotation'].values, labels_df.loc[test_index, 'annotation'].values

        # Creating DataLoader for the current fold
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Model training for the current fold
        model_fold = train_fold(model, train_loader, device, n_epochs)

        # Save the model for the current fold
        torch.save(model_fold, f"{out_model_path}_{str(time.time())}_fold_{fold}.pt")

        predictions, features, labels = evaluate_and_generate_predictions(model_fold, test_loader, device)
        # result_df = pd.DataFrame({
        #     'index': test_index,
        #     'pred': predictions,
        #     'feature': list(embeddings),
        #     'label': labels
        # })
        # result_df.to_csv(f"{out_model_path}_results_fold_{fold}.csv", index=False)
        
        # all_predictions.extend(predictions)
        # all_features.extend(features)
        # all_labels.extend(labels)
    
        # save memory by saving per fold
        np.save(f"{out_model_path}_indices_fold_{fold}.npy", np.array(test_index))
        np.save(f"{out_model_path}_preds_fold_{fold}.npy", np.array(predictions))
        np.save(f"{out_model_path}_features_fold_{fold}.npy", np.array(features))
        np.save(f"{out_model_path}_labels_fold_{fold}.npy", np.array(labels))

def train_fold(model, train_loader, device, n_epochs):
    model = copy.deepcopy(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch"):
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            # Forward pass
            outputs = model(inputs)
            labels = labels.to(torch.long)
            loss = criterion(outputs, labels)

			# Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        torch.cuda.empty_cache()

    return model

def evaluate_and_generate_predictions(model, test_loader, device):
    model.eval()
    predictions = []
    embeddings = []
    labels_list = []
    labels_pred = []

    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            # outputs = model(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            # embedding = model.embedding_layer(inputs)

            outputs, features = model.get_predictions_and_features(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

            label_pred = outputs.argmax(dim=1).detach().cpu().numpy()
            labels_pred.extend(label_pred)
            predictions.extend(outputs.detach().cpu().numpy())
            embeddings.extend(features.detach().cpu().numpy())
            labels_list.extend(labels.detach().cpu().numpy())

    average_loss = total_loss / total_samples
    accuracy = accuracy_score(labels_list, labels_pred)
    precision = precision_score(labels_list, labels_pred, average='weighted') # weighted by class representation in dataset
    recall = recall_score(labels_list, labels_pred, average='weighted')
    cm = confusion_matrix(labels_list, labels_pred)

    print(f"Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("----------------------------------")

    torch.cuda.empty_cache()

    return predictions, embeddings, labels_list

# ------------

# input paths
ims_path = 'tests/data/combined_images_parasite_and_non-parasite.npy' # path to images for training
label_path = 'tests/data/combined_ann_parasite_and_non-parasite.csv' # path to their respective annotations

images = np.load(ims_path)
labels_df = pd.read_csv(label_path, index_col='index')
# labels = labels_df['annotation'].values
# # test with just 100 images
# images = images[:100,:,:,:]
# labels_df[:100]
print("total images: ", images.shape[0])

# model architecture
model_specs = {'model_name':'resnet34','n_channels':4,'n_filters':64,'n_classes':2,'kernel_size':3,'stride':1,'padding':1, 'batch_size':32}

n_classes_derived = labels_df['annotation'].nunique() # number of unique annotation classes in dataset
model = models.ResNet(model=model_specs['model_name'],n_channels=model_specs['n_channels'],n_filters=model_specs['n_filters'], n_classes=n_classes_derived,kernel_size=model_specs['kernel_size'],stride=model_specs['stride'],padding=model_specs['padding'])

# train_frac = 0.7
n_epochs = 2
n_folds = 5
model_out_path = 'tests/cleanlab/outputs/' + model_specs['model_name']
train_cv(model, images, labels_df, model_out_path, n_folds, model_specs['batch_size'], n_epochs)