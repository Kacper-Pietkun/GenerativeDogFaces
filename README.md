# Generative Dog Faces

## Download and preprocess dataset:
Run `prepare_dataset.py` script to:
- download datasets:
    - Stanford Dogs Dataset (vision.stanford.edu/aditya86/ImageNetDogs/images.tar)
    - Kaggle Cats and Dogs Dataset (cats images are deleted from the dataset) (download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)
- Unzip downloaded datasets
- Merge them into one big dataset of dogs' images
- Crop faces of dogs from the images (new file is created for each detected face of a dog). Dog face detector is downloaded from github.com/kairess/dog_face_detector
- Preprocess images:
    - resize to 64x64 pixels
    - convert to torch.Tensor
- Split dataset into train, validation and test sets with ratio 90/5/5%
- Save sets as numpy ".npy" files

When running script, you have to specify destination directory, where created sets will be saved by setting `--destination` parameter. Make sure to create separate directory for that, because at the end of the script its content will be cleared, as there are many intermediate files created during preparation of the dataset.



# links
- https://arxiv.org/pdf/1511.06434v2
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- https://github.com/soumith/ganhacks
- https://arxiv.org/pdf/1312.6114