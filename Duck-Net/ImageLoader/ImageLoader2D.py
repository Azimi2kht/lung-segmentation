import glob

import numpy as np
from PIL import Image
from skimage.io import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

dataset_folder_path = "/home/azimi2kht/projects/chest/RSUA Chest X-Ray Dataset/Data Chest X-Ray RSUA (Validated)/Covid/"  # Add the path to your data directory


def load_data(img_height, img_width, images_to_be_loaded, dataset_name):
    IMAGES_PATH = dataset_folder_path + dataset_name + 'images/'
    MASKS_PATH = dataset_folder_path + dataset_name + 'masks/'

    print(IMAGES_PATH)

    train_ids = glob.glob(IMAGES_PATH + "*.bmp")

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    X_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print('Resizing training images and masks: ' + str(images_to_be_loaded))
    for n, id_ in tqdm(enumerate(train_ids)):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks/").replace("Images", "Mask")

        image = imread(image_path)
        mask_ = imread(mask_path)

        mask = np.zeros((img_height, img_width), dtype=np.bool_)

        pillow_image = Image.fromarray(image)

        pillow_image = pillow_image.resize((img_height, img_width))

        image = np.array(pillow_image)

        X_train[n] = image / 255

        pillow_mask = Image.fromarray(mask_)
        pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
        
        mask_ = np.array(pillow_mask)

        mask_threshold = 127
        mask[:, :] = np.where(mask_[:, :] >= mask_threshold, 1, 0)

        Y_train[n] = mask

    Y_train = np.expand_dims(Y_train, axis=-1)

    return X_train, Y_train
