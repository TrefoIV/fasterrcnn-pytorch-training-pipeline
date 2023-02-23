import torch
import cv2
import numpy as np
import os
import glob as glob
import random
from copy import copy

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils.transforms import (
    get_train_transform, 
    get_valid_transform,
    get_train_aug,
    transform_mosaic
)


# the dataset class
class MatrixDataset(Dataset):
    def __init__(
        self, 
        images_path, 
        labels_path, 
        img_size, 
        classes, 
        transforms=None, 
        use_train_aug=False,
        train=False, 
        no_mosaic=False,
        square_training=False,
        discard_negative_example = True
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.classes = classes
        self.train = train
        self.no_mosaic = no_mosaic
        self.square_training = square_training
        self.all_images = []

        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))

        self.read_and_clean(discard_negative_example)
        
    def read_and_clean(self, discard_negative_example):
        # Discard any images and labels when the XML 
        # file does not contain any object, if negative example are not given
        def check_path_and_save_image(path):
            tree = et.parse(path)
            root = tree.getroot()
            image_name = root.findtext("filename")
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
            #If i don't discard xml than save the image
            if not discard_path:
                self.all_images.append(image_name)
            return discard_path

        self.all_annot_paths = list(filter(check_path_and_save_image, self.all_annot_paths ))

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image.
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self.resize(image, square=self.square_training)
        image_resized /= 255.0
        
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Get the height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]
                
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

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, ymin, xmax, ymax, image_width, image_height
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])
            
            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = (xmin/image_width)*image_resized.shape[1]
            xmax_final = (xmax/image_width)*image_resized.shape[1]
            ymin_final = (ymin/image_height)*image_resized.shape[0]
            ymax_final = (ymax/image_height)*image_resized.shape[0]
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(self, xmin, ymin, xmax, ymax, width, height):
        """
        Check that all x_max and y_max are not more than the image
        width or height.
        """
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        return xmin, ymin, xmax, ymax

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        if self.no_mosaic:
            image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx
            )

        if self.train and not self.no_mosaic:
            #while True:
            image_resized, boxes, labels, \
                area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                idx, resize_factor=(self.img_size, self.img_size)
            )
                # Only needed if we don't allow training without target bounding boxes
               # if len(boxes) > 0:
               #     break
        
        # visualize_mosaic_images(boxes, labels, image_resized, self.classes)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        sample = self.transforms(image=image_resized,
                                    bboxes=target['boxes'],
                                    labels=labels)
        image_resized = sample['image']
        target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)

        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

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
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True, 
        no_mosaic=no_mosaic,
        square_training=square_training,
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
        get_valid_transform(),
        train=False, 
        no_mosaic=True,
        square_training=square_training,
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