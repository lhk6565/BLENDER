from collections import defaultdict
from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import gdown
import uuid
import json
import os

#from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
#from wilds.datasets.fmow_dataset import FMoWDataset 


# utils #######################################################################

def stage_path(data_dir, name): 
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path

def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

def exists_download(data_dir, name):
    if not os.path.exists(data_dir):
        return False
    
    if len(os.listdir(data_dir)) > 0:
        print(f"[INFO] {name} dataset already exists at {data_dir}. Skipping download.")
        return True
    
    return False


# VLCS ########################################################################

# Slower, but builds dataset from the original sources
#
# def download_vlcs(data_dir):
#     full_path = stage_path(data_dir, "VLCS")
# 
#     tmp_path = os.path.join(full_path, "tmp/")
#     if not os.path.exists(tmp_path):
#         os.makedirs(tmp_path)
# 
#     with open("domainbed/misc/vlcs_files.txt", "r") as f:
#         lines = f.readlines()
#         files = [line.strip().split() for line in lines]
# 
#     download_and_extract("http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
#                          os.path.join(tmp_path, "voc2007_trainval.tar"))
#     
#     download_and_extract("https://drive.google.com/uc?id=1I8ydxaAQunz9R_qFFdBFtw6rFTUW9goz",
#                          os.path.join(tmp_path, "caltech101.tar.gz"))
#     
#     download_and_extract("http://groups.csail.mit.edu/vision/Hcontext/data/sun09_hcontext.tar",
#                          os.path.join(tmp_path, "sun09_hcontext.tar"))
#     
#     tar = tarfile.open(os.path.join(tmp_path, "sun09.tar"), "r:")
#     tar.extractall(tmp_path)
#     tar.close()
# 
#     for src, dst in files:
#         class_folder = os.path.join(data_dir, dst)
# 
#         if not os.path.exists(class_folder):
#             os.makedirs(class_folder)
# 
#         dst = os.path.join(class_folder, uuid.uuid4().hex + ".jpg")
# 
#         if "labelme" in src:
#             # download labelme from the web 
#             gdown.download(src, dst, quiet=False)
#         else:
#             src = os.path.join(tmp_path, src)
#             shutil.copyfile(src, dst)
# 
#     shutil.rmtree(tmp_path)


def download_vlcs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "VLCS")
    if exists_download(full_path, "VLCS"):
        return

    download_and_extract("https://drive.google.com/uc?id=15bipndapQ1fRhhgGQx-X3p9-wqJIJWab",
                         os.path.join(full_path, "VLCS.zip"))


# MNIST #######################################################################

def download_mnist(data_dir):
    # Original URL: http://yann.lecun.com/exdb/mnist/
    if exists_download(os.path.join(data_dir, "MNIST"), "MNIST"):
        return
    
    MNIST(data_dir, download=True)


# PACS ########################################################################

def download_pacs(data_dir):
    # Original URL: https://dali-dl.github.io/project_iccv2017.html
    if exists_download(os.path.join(data_dir, "PACS"), "PACS"):
        return

    download_and_extract("https://drive.google.com/uc?id=1SDDAQ2ehn3lL5XAtBPjiHSgeqGoofs7I",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              os.path.join(data_dir, "PACS"))


# Office-Home #################################################################

def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    if exists_download(os.path.join(data_dir, "OfficeHome"), "OfficeHome"):
        return

    download_and_extract("https://drive.google.com/uc?id=1dCZz5OdNauUvi-u4RfzRYtmsFvz9FEBz",
                         os.path.join(data_dir, "office_home.zip"))
    
    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              os.path.join(data_dir, "OfficeHome"))


# DomainNET ###################################################################

def download_domain_net(data_dir):
    # Original URL: http://ai.bu.edu/M3SDA/
    full_path = stage_path(data_dir, "DomainNet")
    if exists_download(full_path, "DomainNet"):
        return

    urls = {"clipart": "https://drive.google.com/uc?id=1UuiJ_WqFH04ibVXn4YG3tpK7MJEiihiW",
            "infograph": "https://drive.google.com/uc?id=1enp9_1D1vo28B-p-LEi-F8NvFku_ZVCb",
            "painting": "https://drive.google.com/uc?id=15PCTN17Q2qLE_tz7nYLtpDItPYhto5BS",
            "quickdraw": "https://drive.google.com/uc?id=1g2xCfSns394LFQUtDxHA7drJTyKBZcUT",
            "real": "https://drive.google.com/uc?id=1X_KfOq9bmyrQ71kXmhbrWAOkPtNTNntT",
            "sketch": "https://drive.google.com/uc?id=1uhCzKKVClqzvdHsNCXBG6gpP9-mywarh"}
    
    for domain, url in urls.items():
        download_and_extract(url, os.path.join(full_path, f"{domain}.zip"))

    download_and_extract("https://drive.google.com/uc?id=14tmS_VaUHoflJsaeVGQeNEojxFMsqJbx",
                         os.path.join(full_path, "DomainNet_duplicates.txt"), remove=False)
   
    with open(os.path.join(full_path, "DomainNet_duplicates.txt"), "r") as f:
        for line in f.readlines():
            try:
                os.remove(os.path.join(full_path, line.strip()))
            except OSError:
                pass


# TerraIncognita ##############################################################

def download_terra_incognita(data_dir):
    # Original URL: https://beerys.github.io/CaltechCameraTraps/
    # New URL: http://lila.science/datasets/caltech-camera-traps

    full_path = stage_path(data_dir, "TerraIncognita")
    if exists_download(full_path, "TerraIncognita"):
        return
    
    download_and_extract("https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
                         os.path.join(full_path, "terra_incognita_images.tar.gz"))
    
    download_and_extract("https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz",
                         os.path.join(full_path, "terra_incognita_annotations.tar.gz"))

    include_locations = [38, 46, 100, 43]

    include_categories = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
                          "raccoon", "squirrel"]

    images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
    annotations_folder = os.path.join(full_path,"eccv_18_annotation_files/")
    cis_test_annotations_file = os.path.join(full_path, "eccv_18_annotation_files/cis_test_annotations.json")
    cis_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/cis_val_annotations.json")
    train_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/train_annotations.json")
    trans_test_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_test_annotations.json")
    trans_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_val_annotations.json")
    annotations_file_list = [cis_test_annotations_file, cis_val_annotations_file, train_annotations_file, trans_test_annotations_file, trans_val_annotations_file]
    destination_folder = full_path

    stats = {}
    data = defaultdict(list)

    for annotations_file in annotations_file_list:
        annots = {}
        with open(annotations_file, "r") as f:
            annots = json.load(f)
            for k, v in annots.items():
                data[k].extend(v)

    category_dict = {}
    for item in data['categories']:
        category_dict[item['id']] = item['name']

    for image in data['images']:
        image_location = image['location']

        if image_location not in include_locations:
            continue

        loc_folder = os.path.join(destination_folder,
                                  'location_' + str(image_location) + '/')

        if not os.path.exists(loc_folder):
            os.mkdir(loc_folder)

        image_id = image['id']
        image_fname = image['file_name']

        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                if image_location not in stats:
                    stats[image_location] = {}

                category = category_dict[annotation['category_id']]

                if category not in include_categories:
                    continue

                if category not in stats[image_location]:
                    stats[image_location][category] = 0
                else:
                    stats[image_location][category] += 1

                loc_cat_folder = os.path.join(loc_folder, category + '/')

                if not os.path.exists(loc_cat_folder):
                    os.mkdir(loc_cat_folder)

                dst_path = os.path.join(loc_cat_folder, image_fname)
                src_path = os.path.join(images_folder, image_fname)

                shutil.copyfile(src_path, dst_path)

    shutil.rmtree(images_folder)
    shutil.rmtree(annotations_folder)


# SVIRO #################################################################

def download_sviro(data_dir):
    # Original URL: https://sviro.kl.dfki.de
    full_path = stage_path(data_dir, "sviro")
    
    download_and_extract("https://sviro.kl.dfki.de/?wpdmdl=1731", 
                         os.path.join(data_dir, "sviro_grayscale_rectangle_classification.zip"))

    os.rename(os.path.join(data_dir, "SVIRO_DOMAINBED"), 
              full_path)


if __name__ == "__main__":
    root = './datasets'

    download_mnist(root)
    download_pacs(root)
    download_office_home(root)
    download_domain_net(root)
    download_vlcs(root)
    download_terra_incognita(root)
    # download_sviro(args.data_dir)
    # Camelyon17Dataset(root_dir=args.data_dir, download=True)
    # FMoWDataset(root_dir=args.data_dir, download=True)