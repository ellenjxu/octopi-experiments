import os
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
    
    def extract_early_features(self, x):  # try earlier layer
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)

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
            images =     batch[0].to(device)
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
# data_id = ['parasite', 'negative', 'unsure']

n_max = 10000

# get data

def get_images_threshold(csv_path, np_path, threshold):
    images = []

    np_array = np.load(np_path)
    df = pd.read_csv(csv_path)

    # threshold for positive parasites
    filtered_df = df[df['parasite output'] > threshold]

    # get the corresponding image at the index
    for index in filtered_df['index']:
        images.append(np_array[index])

    return images

def get_images_from_csv(csv_dir, npy_dir, threshold=0.85):
    data = []
    slide_names = []

    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv') and not csv_file.startswith('SBC'):
            csv_path = os.path.join(csv_dir, csv_file)
            # out_path = os.path.join(out_dir, csv_file.replace('.csv', '.npy'))
            np_path = os.path.join(npy_dir, csv_file.replace('.csv', '.npy'))

            # get the images
            images = get_images_threshold(csv_path, np_path, threshold)
            # get random 1000
            images = random.sample(images, 1000)
            data.extend(images)
            slide_names.extend([csv_file.replace('.csv', '')] * len(images))

    return data, slide_names

csv_dir = 'model_output/'
npy_dir = 'npy_v2/'
out_dir = 'tests/visualization/data/'
threshold = 0.85

### for pos umap
# data, slide_names = get_images_from_csv(csv_dir, npy_dir)
# data = np.array(data)/255.0
# slide_names = np.array(slide_names)

### for neg and unsure umap
neg_data = np.load('tests/visualization/negative_unsure 0.5.npy', allow_pickle=True)
unsure_data = np.load('tests/visualization/unsure_unsure 0.8.npy', allow_pickle=True)
unsure_data = np.array(random.sample(list(unsure_data), 1000))

data = np.concatenate((neg_data, unsure_data), axis=0)
data = np.array(data)/255.0
slide_names = np.concatenate((np.array(['negative']*len(neg_data)), np.array(['unsure']*len(unsure_data))), axis=0)

# limit the max number of points
# n_max = min(10000, len(data))
# sample_indices = np.random.choice(len(data), size=n_max, replace=False)
# data = data[sample_indices]
# slide_names = slide_names[sample_indices]

label = np.ones(len(data))
print('data shape: ', data.shape)
print('label shape: ', label.shape)

# ============================================== #
# ============================================== #
# combine
X = data
y = label
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

print('features shape: ', features.shape)
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

# Map each unique slide name to a color
unique_slides = np.unique(slide_names)
print('unique slides: ', len(unique_slides))
colors = sns.color_palette("hsv", len(unique_slides))
color_map = dict(zip(unique_slides, colors))

# Assign colors to each data point based on its slide name
color_labels = [color_map[slide] for slide in slide_names]

# Apply UMAP
print('applying UMAP')
reducer = umap.UMAP(n_components=2)
# reducer = umap.UMAP(n_neighbors=30,
#                     min_dist=0.1,
#                     n_components=2,
#                     metric='euclidean',
#                     learning_rate=1.0,
#                     n_epochs=500,
#                     spread=1.0,
#                     random_state=42)
embedding = reducer.fit_transform(all_features)

# Visualize the data with colors based on slides
sns.set(font_scale=1)
plt.figure(figsize=(20, 12))
for slide, color in color_map.items():
    indices = [i for i, s in enumerate(slide_names) if s == slide] # get the indices of the images from slide s
    sns.scatterplot(x=embedding[indices, 0], y=embedding[indices, 1], color=color, label=slide, s=5)

plt.legend(bbox_to_anchor=(1.25, 1), loc=2, borderaxespad=0., fontsize='x-small')
plt.tight_layout(rect=[0, 0, 1, 0.85])  # left bottom width height
plt.savefig("tests/visualization/" + "umap_unsure_" + str(time.time()) + ".png", dpi=300)
plt.show()