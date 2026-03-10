import os
import cv2
import random
from shutil import rmtree, copyfile


def extract_frames(src_folder, dest_folder, extension='jpg'):
    """
    A function for extracting frames from videos saved in source folder and saving all the extracted images in
    destination folder.


    :param src_folder: A string for the source folder containing videos.

    :param dest_folder: A string for the destination folder where the extracted images will be saved.

    :param extension: The extension to be used to save the extracted images.
    """
    frame_count = 0
    img_num = 0
    for vid_name in os.listdir(src_folder):
        if vid_name.endswith('.mp4'):
            vid_path = os.path.join(src_folder, vid_name)
            cap = cv2.VideoCapture(vid_path)
            success, image = cap.read()

            while success:
                if frame_count % 5 == 0:
                    dest_image_path = os.path.join(dest_folder, str(img_num).zfill(6)+"."+extension)
                    image = cv2.resize(image, (224, 224))
                    cv2.imwrite(dest_image_path, image)
                    img_num += 1

                    if img_num % 10000 == 0 and img_num > 1:
                        print("Saved " + str(img_num) + " images in " + dest_folder)

                success, image = cap.read()
                frame_count += 1

            cap.release()


def train_test_split(src_folder, dest_train, dest_test):
    """
    A function to split a folder containing images into separate folders for train and test sets.
    The dataset is split in the ratio of 80:20.


    :param src_folder: Source folder containing the extracted images.

    :param dest_train: The destination folder where training set images are to be saved.

    :param dest_test: The destination folder where test set images are to be saved.
    """
    os.makedirs(dest_train, exist_ok=True)
    os.makedirs(dest_test, exist_ok=True)
    images = [f for f in os.listdir(src_folder) if f.endswith(".jpg")]

    random.shuffle(images)
    split_index = int(len(images) * 0.8)
    train_images = images[:split_index]
    test_images = images[split_index:]

    for image in train_images:
        src = src_folder + "/" + image
        dest = dest_train + "/" + image

        try:
            copyfile(src, dest)
        except PermissionError:
            print("denied permission")
        except:
            print("Some error occurred")

    print("Done with train for " + src_folder)
    for image in test_images:
        src = src_folder + "/" + image
        dest = dest_test + "/" + image
        try:
            copyfile(src, dest)
        except PermissionError:
            print("denied permission")
        except:
            print("Some error occurred")

    print("Done with test for " + src_folder)


if not os.path.exists("temp"):
    os.makedirs("temp")
    print("Created 'temp' folder")

real_dest_path = "temp/real"
if not os.path.exists(real_dest_path):
    os.makedirs(real_dest_path)
    print("Created 'real' folder")

extract_frames("dataset/original_sequences/youtube/c23/videos", real_dest_path)
print("Saved all the images in real folder")


fake_dest_path = "temp/fake"
if not os.path.exists(fake_dest_path):
    os.makedirs(fake_dest_path)
    print("Created 'fake' folder")

fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
for fake_type in fake_types:
    type_dest_path = fake_dest_path + "/" + fake_type
    if not os.path.exists(type_dest_path):
        os.makedirs(type_dest_path)
        print("Created " + type_dest_path + " folder")

    extract_frames("dataset/manipulated_sequences/c23/videos" + fake_type, type_dest_path)
    print("Saved all images in " + type_dest_path + " folder")

if not os.path.exists("data"):
    os.makedirs("data")
    print("Created 'data' folder")

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)
train_test_split("temp/real", "data/train/real", "data/test/real")
print("Done with real")

for fake in fake_types:
    fake_src = "temp/fake/" + fake
    fake_dest_train = "data/train/" + fake
    fake_dest_test = "data/test/" + fake

    train_test_split(fake_src, fake_dest_train, fake_dest_test)
    print("Done with " + fake)

if os.path.exists('./temp'):
    rmtree('./temp')


print("All done!")
