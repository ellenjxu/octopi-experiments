# inputs:
# - `model_output/` contains .csv files with image index and predictions. The relevant column names are 'index' and 'parasite output'
# - `npy_v2/` contains the .npy image files

import os
import pandas as pd
import numpy as np

def get_pos_images(csv_dir, npy_dir, threshold):
    """
    loops through corresponding files in input dirs
    creates numpy image arrays with parasite prediction score greater than a threshold
    """
    pos_images = []

    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_dir, csv_file)
            np_array = np.load(os.path.join(npy_dir, csv_file.replace('.csv', '.npy')))
            df = pd.read_csv(csv_path)

            # threshold for positive parasites
            filtered_df = df[df['parasite output'] > threshold]

            # get the corresponding image at the index
            for index in filtered_df['index']:
                pos_images.append(np_array[index])

    return  pos_images

csv_dir = 'model_output/'
npy_dir = 'npy_v2/'
threshold = 0.5

pos_images = get_pos_images(csv_dir, npy_dir, threshold)
np.save('pos_images_thresholded_from_csv.npy', pos_images)