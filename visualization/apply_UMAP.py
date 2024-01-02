import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import time 
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import models
from sklearn.decomposition import PCA

models_dict = {'resnet18': torchvision.models.resnet18,
               'resnet34': torchvision.models.resnet34,
               'resnet50': torchvision.models.resnet50,
               'resnet101': torchvision.models.resnet101,
               'resnet152': torchvision.models.resnet152}

class ResNet(nn.Module):
    def __init__(self, model='resnet18',n_channels=4,n_filters=64,n_classes=1,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.n_classes = n_classes
        self.base_model = models_dict[model](pretrained=True)
        self._feature_vector_dimension = self.base_model.fc.in_features
        self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
        self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        return self.fc(features)

    def extract_features(self,x):
        x = self.base_model(x)
        return x.view(x.size(0), -1)

    def get_predictions(self,x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        output = self.fc(features)
        if self.n_classes == 1:
            return torch.sigmoid(output)
        else:
            return torch.softmax(output,dim=1)

    def get_predictions_and_features(self,x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        output = self.fc(features)
        if self.n_classes == 1:
            return torch.sigmoid(output), features
        else:
            return torch.softmax(output,dim=1), features

    def get_features(self,x):
        x = self.base_model(x)
        features = x.view(x.size(0), -1)
        return features

# Function to extract embeddings using CUDA
def get_embeddings(data_loader, model, device):
    with torch.no_grad():
        embeddings = []
        for batch in data_loader:
            images = batch[0].to(device)
            features = model.get_features(images)
            embeddings.append(features.cpu())
        return torch.cat(embeddings, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(model='resnet34', n_channels=4, n_filters=64, n_classes=3, kernel_size=3, stride=1, padding=1)
model.load_state_dict(torch.load('tests/visualization/model_resnet34_1704164366.7755363.pt'))
model = model.to(device) 
model.eval()

# ============================================== #
# ============================================== #
data_id = ['parasite', 'negative', 'unsure']
data = {}
label = {}

n_max = 10000

# read data
for i in range(len(data_id)):

    print(data_id[i])

    data[i] = np.load('../data/' + data_id[i] + '.npy')/255.0
    
    if data_id[i] == 'negative':
        fraction = 0.15
        # sample images
        sample_indices = np.random.choice(len(data[i]), size=int(fraction*len(data[i])), replace=False)
        # sample_indices = np.sort(sample_indices)
        data[i] = data[i][sample_indices]

    # limit the max number of points
    sample_indices = np.random.choice(len(data[i]), size=min(len(data[i]),n_max), replace=False)
    data[i] = data[i][sample_indices]

    label[i] = i*np.ones(data[i].shape[0])
    print(data[i].shape)


# ============================================== #
# ============================================== #
# combine
X = np.concatenate([data[i] for i in range(len(data_id))],axis=0)
y = np.concatenate([label[i] for i in range(len(data_id))],axis=0)
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

batch_size = 1024
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

all_features = []
all_labels = []
for i, (images, labels) in enumerate(dataloader):
    images = images.float().cuda()
    features = model.get_features(images)
    features = features.detach().cpu().numpy()
    features = features.squeeze()
    all_features.append(features)
    all_labels.append(labels.detach().cpu().numpy())

all_features = np.vstack(all_features)
all_labels = np.hstack(all_labels)


# ============================================== #
# ============================================== #
# # Initialize UMAP for 3D reduction
# umap_3d = umap.UMAP(n_components=3, random_state=42)

# # Fit the model and transform your data to 3 components
# embedding_3d = umap_3d.fit_transform(all_features)

# # Extract the three components
# x, y, z = embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2]

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(x, y, z, c=all_labels, cmap='Spectral')
# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)
# plt.show()

'''
map_2d = umap.UMAP(n_components=2, random_state=42)
# map_2d = PCA(n_components=2)

# Fit the model and transform your data to 2 components
embedding_2d = map_2d.fit_transform(all_features)

# Extract the two components
x, y = embedding_2d[:, 0], embedding_2d[:, 1]

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(x, y, c=all_labels, cmap='Spectral', s=5)
plt.colorbar(label='Labels')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('2D UMAP Visualization of Features')
plt.savefig("umap_" + str(time.time()) + ".png", dpi=300)
plt.show()
'''


# Apply UMAP
print('applying UMAP')
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(all_features)

# Visualize the data
import matplotlib
# current_palette = matplotlib.colors.hex2color('#86b92e')
# color_dict = dict({'0-9':'#787878'})
#sns.scatterplot(x=embedding[:,0], y=embedding[:,1], edgecolor = 'none', hue=all_labels, legend='full', palette='rainbow')
sns.set(font_scale=1.2)
# color_dict = dict({0:'#787878'})
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], edgecolor = 'none', hue=all_labels[:].astype(int), legend='full', s = 5)
# sns.color_palette("tab10", as_cmap=True)
plt.savefig("umap_" + str(time.time()) + ".png", dpi=300)
plt.show()