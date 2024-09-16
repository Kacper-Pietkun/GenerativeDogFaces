import os
import cv2
import dlib
import shutil
import zipfile
import tarfile
import requests
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


parser = ArgumentParser("Download data and prepare splitted dataset into train, validation and test sets")

parser.add_argument("-d", "--destination", type=str, required=True,
                    help="Destination folder where train, validation, test sets will be saved")


def download_datasets(args, datasets_urls):
    print("Downloading datasets...")
    paths = []
    for idx, (url, images_folder) in enumerate(datasets_urls):
        extension = os.path.splitext(url)[1]
        save_path = os.path.join(args.destination, f"dataset_{idx}{extension}")
        download_from_url(url, save_path)
        paths.append((save_path, images_folder))

    return paths


def download_from_url(url, destination):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(destination, 'wb') as file:
                for chunk in tqdm(response.iter_content(chunk_size=1024)):
                    file.write(chunk)
                    progress_bar.update(len(chunk))

            print(f"'{url}' downloaded successfully")
        else:
            raise Exception(f"Failed to download '{url}'. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred when downloading '{url}'")
        raise


def unzip_datasets(args, paths):
    print("Unzipping datasets...")
    saved_paths = []
    for (path, images_folder) in paths:
        extension = os.path.splitext(path)[1]
        save_path = args.destination
        if extension == ".zip":
            unzip_file(path, save_path)
        elif extension == ".tar":
            untar_file(path, save_path)
        else:
            print(f"Uncrecognized extension for dataset in {path} - skipping")
            continue
        saved_paths.append(os.path.join(save_path, images_folder))
    return saved_paths


def unzip_file(zip_file, destination):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)


def untar_file(tar_file, destination):
    with tarfile.open(tar_file, 'r') as tar:
        tar.extractall(destination)


def merge_datasets(args, paths):
    print("Merging datasets...")
    path_merged = os.path.join(args.destination, "merged")
    if os.path.exists(path_merged):
        shutil.rmtree(path_merged)
    os.makedirs(path_merged)

    counter = 1
    for path in tqdm(paths, desc="Dataset Loop"):
        for root, _, files in tqdm(os.walk(path), desc="Directories Loop"):
            for file_name in tqdm(files, desc="files Loop", leave=False):
                _, extension = os.path.splitext(file_name)
                file_path = os.path.join(root, file_name)
                new_path = os.path.join(path_merged, f"{counter}{extension}")
                os.rename(file_path, new_path)
                counter += 1
    return path_merged


def crop_dog_faces(args, dataset_path):
    print("Cropping dogs faces...")

    dog_head_detector_model_path = os.path.join(args.destination, "detector.dat")
    download_from_url("https://github.com/kairess/dog_face_detector/raw/master/dogHeadDetector.dat",
                      dog_head_detector_model_path)
    detector = dlib.cnn_face_detection_model_v1(dog_head_detector_model_path)
    save_cropped_faces_path = os.path.join(args.destination, "cropped_faces", "dogs")
    os.makedirs(save_cropped_faces_path)
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files):
            try:
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                os.remove(file_path)

                outputs = detector(img, upsample_num_times=1)
                for i, output in enumerate(outputs):
                    if output.confidence <= 0.8:
                        continue

                    crop_img = img[output.rect.top():output.rect.bottom(), output.rect.left():output.rect.right()]
                    save_path = os.path.join(save_cropped_faces_path, f"cropped_{i}_{file}")
                    cv2.imwrite(save_path, crop_img)
            except:
                pass
    return os.path.dirname(save_cropped_faces_path)


def preprocess_images(dataset_path):
    print("Preprocessing images...")
    transform_with_sharpness = transforms.Compose([transforms.Grayscale(),
                                                   transforms.Resize((64, 64)),
                                                   transforms.RandomAdjustSharpness(sharpness_factor=3, p=1.0),
                                                   transforms.ToTensor()])
    dataset = datasets.ImageFolder(dataset_path, transform=transform_with_sharpness)
    dataset_tensor = torch.stack([img for img, _ in dataset])
    return dataset_tensor


def split_dataset(dataset):
    print("Splitting dataset...")
    train_ratio = 0.9
    train_images, validation_images = train_test_split(dataset, train_size=train_ratio, random_state=42)
    validation_images, test_images = train_test_split(validation_images, train_size=0.5, random_state=42)
    return train_images, validation_images, test_images


def clean_files(args):
    print("Cleaning files...")
    user_input = ""
    while user_input != "y" and user_input != "n":
        user_input = input(f"Do you agree to clean the directory specified by 'args.destination' "
                        f"path - {args.destination}? ! BEWARE, THIS WILL DELETE ALL THE FILES IN "
                        f"THIS DIRECTORY, EXCEPT THE PROCESSED DATASET [y/n]:")
        if user_input == "y":
            shutil.rmtree(args.destination)
            os.makedirs(args.destination)
            print("Files Deleted...")
        else:
            print("Skip cleaning files...")


def save_splits(args, splits):
    print("Saving train, validation and test sets...")
    for (split, split_name) in splits:
        save_path = os.path.join(args.destination, f"{split_name}.npy")
        np.save(save_path, split)
        

def main():
    args = parser.parse_args()

    datasets_urls = [("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip", "PetImages/Dog"),
                     ("http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", "Images")]

    paths = download_datasets(args, datasets_urls)
    paths = unzip_datasets(args, paths)
    path = merge_datasets(args, paths)
    path =  crop_dog_faces(args, path)
    dataset = preprocess_images(path)
    train_images, validation_images, test_images = split_dataset(dataset)
    clean_files(args)
    save_splits(args, [(train_images, "train_images"),
                       (validation_images, "validation_images"),
                       (test_images, "test_images")])


if __name__ == "__main__":
    main()
