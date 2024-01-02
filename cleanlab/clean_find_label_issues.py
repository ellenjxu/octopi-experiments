import torch
import torch.nn as nn
import torchvision
from skorch import NeuralNetClassifier
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split

from cleanlab import Datalab

# https://cleanlab.ai/blog/label-errors-image-datasets/

# Define your ResNet class
class ResNet(nn.Module):
    def __init__(self, model='resnet18', n_channels=4, n_filters=64, n_classes=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.n_classes = n_classes
        models_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152
        }
        self.base_model = models_dict[model](pretrained=False)
        self._feature_vector_dimension = self.base_model.fc.in_features
        self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
        self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        x = self.fc(features)
        return x

# Load the data
parasite_images = np.load('data/parasite.npy')
negative_images = np.load('data/negative.npy')
unsure_images = np.load('data/unsure.npy')

# Create labels
parasite_labels = np.ones(len(parasite_images))  # Label 1 for parasites
negative_labels = np.zeros(len(negative_images))  # Label 0 for non-parasites
unsure_labels = 2*np.ones(len(unsure_images))  # Label 0 for non-parasites

# Combine the datasets
images = np.concatenate((parasite_images, negative_images, unsure_images), axis=0)
labels = np.concatenate((parasite_labels, negative_labels, unsure_labels), axis=0)
labels = labels.astype("int64")

# Sample the data
fraction = 0.05
sample_indices = np.random.choice(len(images), size=int(fraction*len(images)), replace=False)
# sample_indices = np.sort(sample_indices)
images = images[sample_indices]
labels = labels[sample_indices]

# Save the sampled images
id_ = str(time.time())
np.save('sampled_images_'+id_+'.npy', images)
labels_df = pd.DataFrame(labels, columns=['annotation'])
labels_df.to_csv('sampled_labels_'+id_+'.csv', index=True)

# format the image
images = images/255.0
images = images.astype("float32")

# model
model = 'resnet34'
max_epochs = 5

# Define the neural network using skorch
net = NeuralNetClassifier(
    module=ResNet,
    module__model=model,  # Choose the ResNet model you want
    module__n_channels=4,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    lr=0.001,
    batch_size=32,
    max_epochs=max_epochs,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
)

# load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# out of sample
from sklearn.model_selection import cross_val_predict
num_crossval_folds = 5
pred_probs = cross_val_predict(net, images, labels,
                               cv=num_crossval_folds,
                               method='predict_proba')

from sklearn.metrics import accuracy_score
predicted_labels = pred_probs.argmax(axis=1)
acc = accuracy_score(labels, predicted_labels)
print(f"Cross-validated estimate of accuracy on held-out data: {acc}")


from cleanlab.filter import find_label_issues
ranked_label_issues = find_label_issues(labels, pred_probs, return_indices_ranked_by="self_confidence")
print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
print("Here are the indices of the top 15 most likely label errors:\n"
      f"{ranked_label_issues[:15]}")

import matplotlib.pyplot as plt

def plot_examples(id_iter, nrows=1, ncols=1):
    for count, id in enumerate(id_iter):

        frame = images[id]
        frame = frame.transpose(1,2,0)
        img_fluorescence = frame[:,:,[2,1,0]]
        img_dpc = frame[:,:,3]
        img_dpc = np.dstack([img_dpc,img_dpc,img_dpc])
        img_overlay = 0.64*img_fluorescence + 0.36*img_dpc

        plt.subplot(nrows, ncols, count + 1)
        plt.imshow(img_overlay)
        plt.title(f"id: {id} \n label: {labels[id]}")
        plt.axis("off")

    plt.tight_layout(h_pad=2.0)
    plt.show()

plot_examples(ranked_label_issues[range(4*6)], 4, 6)

print(ranked_label_issues)

# save the result
np.save(str(len(ranked_label_issues)) + ' possible label issues.npy',images[ranked_label_issues])
labels_df.to_csv('sampled_labels.csv', index=False)

# Extract and save labels corresponding to images with possible label issues
labels_with_issues = labels[ranked_label_issues]
labels_with_issues_df = pd.DataFrame(labels_with_issues, columns=['annotation'])
labels_with_issues_df.to_csv(str(len(ranked_label_issues)) + '_possible_label_issues_labels_' + id_ + '.csv', index=True)