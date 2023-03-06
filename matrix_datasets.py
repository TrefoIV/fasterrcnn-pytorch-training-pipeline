import math
from typing import Dict, Union
from xml.etree.ElementTree import ParseError

import torch
import cv2
import numpy as np
import os
import glob as glob
import random
import csv
from copy import copy

from xml.etree import ElementTree as et

from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# the dataset class
class MatrixDataset(Dataset):
    def __init__(
            self,
            images_path,
            labels_path,
            img_size,
            classes,
            transforms=None,
            train=False,
            discard_negative_example=True
    ):
        self.transforms = transforms
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train

        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))

        self.read_and_clean(discard_negative_example)

    def read_and_clean(self, discard_negative_example):
        # Discard any images and labels when the XML 
        # file does not contain any object, if negative example are not given
        global invalid_xml
        invalid_xml = 0

        def check_path_and_save_image(path):
            try:
                tree = et.parse(path)
            except ParseError:
                global invalid_xml
                invalid_xml += 1
                return False

            root = tree.getroot()
            image_name: str = root.findtext("filename")
            image_path = os.path.join(self.images_path, image_name)
            discard_path = False
            if not os.path.exists(image_path):
                print(f"Matrix {image_path} associated to {path} not found...")
                print(f"Discarding {path}...")
                discard_path = True

            if not discard_path and discard_negative_example:
                object_present = False
                for member in root.findall('object'):
                    if member.find('bndbox'):
                        object_present = True
                        break
                if not object_present:
                    print(f"File {path} contains no object. Discarding xml file and image...")
                    discard_path = True
            return not discard_path

        self.all_annot_paths = list(filter(check_path_and_save_image, self.all_annot_paths))
        print("XML INVALIDI: " + str(invalid_xml))

    def read_matrix_file(self, path: str) -> Tensor:
        reader = csv.reader(open(path, "r"), delimiter=",")
        x = list(reader)
        result = np.array([x]).astype(np.float32)
        max_value: float = math.pow(2, (2 * result.shape[1]))
        result = result / max_value
        return torch.from_numpy(result)

    def load_image_and_labels(self, index):
        annot_filename = self.all_annot_paths[index]

        tree = et.parse(annot_filename)
        root = tree.getroot()

        image_name: str = root.findtext("filename")
        image_path = os.path.join(self.images_path, image_name)

        # Read the image.
        image = self.read_matrix_file(image_path)

        # Get the height and width of the image.
        image_width: int = image.shape[1]
        image_height: int = image.shape[0]

        boxes = []
        orig_boxes = []
        labels = []

        # Box coordinates for xml files are extracted and corrected for image size given.
        for member in root.findall('object'):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(float(member.find('bndbox').find('xmin').text))
            # xmax = right corner x-coordinates
            xmax = int(float(member.find('bndbox').find('xmax').text))
            # ymin = left corner y-coordinates
            ymin = int(float(member.find('bndbox').find('ymin').text))
            # ymax = right corner y-coordinates
            ymax = int(float(member.find('bndbox').find('ymax').text))

            orig_boxes.append([xmin, ymin, xmax, ymax])

            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = xmin.shape[1]
            xmax_final = xmax.shape[1]
            ymin_final = ymin.shape[0]
            ymax_final = ymax.shape[0]

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.int32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes,
                                                                                                                  dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes,
                                                                                                             dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int32)
        return image, orig_boxes, boxes, labels, area, iscrowd, (image_width, image_height)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, np.ndarray]]:
        image, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
            index=idx
        )

        # Prepare the final `target` dictionary.
        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image, target

    def __len__(self):
        return len(self.all_annot_paths)


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# Prepare the final datasets and data loaders.
def create_train_dataset(
        train_dir_images,
        train_dir_labels,
        img_size,
        classes,
        use_train_aug=False,
        no_mosaic=False,
        square_training=False,
        discard_negative=True
):
    train_dataset = MatrixDataset(
        train_dir_images,
        train_dir_labels,
        img_size,
        classes,
        train=True,
        discard_negative_example=discard_negative
    )
    return train_dataset


def create_valid_dataset(
        valid_dir_images,
        valid_dir_labels,
        img_size,
        classes,
        square_training=False,
        discard_negative=True
):
    valid_dataset = MatrixDataset(
        valid_dir_images,
        valid_dir_labels,
        img_size,
        classes,
        train=False,
        discard_negative_example=discard_negative
    )
    return valid_dataset


def create_train_loader(
        train_dataset, batch_size, num_workers=0, batch_sampler=None
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return train_loader


def create_valid_loader(
        valid_dataset, batch_size, num_workers=0, batch_sampler=None
):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler
    )
    return valid_loader
