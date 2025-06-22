import torch
from torchvision.datasets import VOCSegmentation, VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
# from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor
from torchvision.datasets import VisionDataset
import numpy as np
import os
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from PIL import Image

# ImageNet
# !wget https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt
def get_imagenet_class(src='./data/imagenet.txt'):
    with open(src, 'r') as f:
        #lines = f.readlines()
        #lines = list(map(lambda x:x.split(':'), lines))
        #imagenet_class = {int(k.strip()): v.strip()[1:-2] for k, v in lines}
        
        lines = f.read()
        lines = list(map(lambda x:x.split(':')[1], lines[1:].split('\n')[:-1]))
    imagenet_class = [line.strip()[1:-2] for line in lines]

    return imagenet_class
#COCO
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
]

def get_coco_class():
    """
    Trả về danh sách tên lớp COCO theo thứ tự category_id tăng dần.
    """
    return [cat["name"] for cat in COCO_CATEGORIES]

def get_coco_colormap():
    """
    Trả về danh sách mã màu RGB tương ứng với các lớp COCO theo thứ tự category_id tăng dần.
    """
    return [cat["color"] for cat in COCO_CATEGORIES]

# VOC
# VOC class names
voc_class = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
voc_class_num = len(voc_class)

# VOC COLOR MAP
voc_colormap = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

def get_voc_class():
    return voc_class

def get_voc_colormap():
    return voc_colormap


# transformation
voc_mean = [0.485, 0.456, 0.406]
voc_std = [0.229, 0.224, 0.225]
#h,w = 520, 520
#h,w = 256, 256 -> RandomCrop 224

def voc_train_dataset(args, img_list, mode='cls'):
    tfs_train = tfs.Compose([tfs.Resize((args.train['input_size'], args.train['input_size'])),  
                            tfs.RandomHorizontalFlip(),
                            tfs.RandomCrop(args.train['crop_size'], padding=4, padding_mode='reflect'),
                            #tfs.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            tfs.ToTensor(),
                            tfs.Normalize(voc_mean, voc_std),
                            ])
    tfs_target = tfs.Compose([tfs.Resize((args.train['crop_size'], args.train['crop_size']))
                            ])

    if mode == 'cls':
        dataset = VOCClassification(root=args.dataset_root, year='2012', image_set='train', 
                                    dataset_list=img_list, download=False, transform=tfs_train)
    elif mode == 'seg':
        dataset = VOCSegmentationInt(root=args.dataset_root, year='2012', image_set='train', 
                                 download=False, transform=tfs_train, target_transform=tfs_target)

    return dataset

def voc_val_dataset(args, img_list, mode='cls'):
    tfs_val = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size'])),  
                            tfs.ToTensor(),
                            tfs.Normalize(voc_mean, voc_std),
                            ])
    tfs_target = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size']))
                            ])

    if mode == 'cls':
        dataset = VOCClassification(root=args.dataset_root, year='2012', image_set='val', 
                                    dataset_list=img_list, download=False, transform=tfs_val)
    elif mode == 'seg':
        dataset = VOCSegmentationInt(root=args.dataset_root, year='2012', image_set='val',
                                 download=False, transform=tfs_val, target_transform=tfs_target)
    return dataset

def voc_test_dataset(args, img_list, mode='cls'):
    tfs_test = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size'])),
                            tfs.ToTensor(),
                            tfs.Normalize(voc_mean, voc_std),
                            ])
    tfs_target = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size']))
                            ])

    if mode == 'cls':
        dataset = VOCClassification(root=args.dataset_root, year='2012', image_set='test', 
                                    dataset_list=img_list, download=False, transform=tfs_test)
    elif mode == 'seg':
        dataset = VOCSegmentationInt(root=args.dataset_root, year='2012', image_set='test', 
                                 download=False, transform=tfs_test, target_transform=tfs_target)
    return dataset
    

def re_normalize(x, mean=voc_mean, std=voc_std):
    x_r = x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean,std)):
        x_r[c] *= std_c
        x_r[c] += mean_c
    return x_r

class VOCSegmentationInt(VOCSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, seg = super().__getitem__(index)
        #seg = torch.LongTensor(seg)
        seg = np.array(seg, dtype=np.uint8)
        return img, seg


class VOCClassification(VOCDetection):
    def __init__(self, *args, **kwargs):
        # Init
        self.dataset_list = kwargs['dataset_list']
        kwargs.pop('dataset_list', None)

        super(VOCClassification, self).__init__(*args, **kwargs)

        self.voc_class = voc_class
        self.voc_class_num = voc_class_num
        self.voc_colormap = voc_colormap
        
        # directory initialization
        image_dir = os.path.split(self.images[0])[0]
        annotation_dir = os.path.join(os.path.dirname(image_dir), 'Annotations')

        # read list of train_aug
        with open(self.dataset_list, 'r') as f:
            train_aug = f.read().split()
        # replace train into train_aug(images, annotations)
        self.images = [os.path.join(image_dir, x + ".jpg") for x in train_aug]
        
        # deprecated(read-only property)
        #self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in train_aug]
        # Re-append xml file list
        self.annotations.clear()
        for x in train_aug:
            self.annotations.append(os.path.join(annotation_dir, x + ".xml"))

    def __getitem__(self, index):
        img, ann = super().__getitem__(index)
        
        # get object list
        objects = ann['annotation']['object']
        # get unique classes
        ann = torch.LongTensor(list({self.voc_class.index(o['name'])-1 for o in objects}))
        # make one-hot encoding
        one_hot = torch.zeros(self.voc_class_num-1)
        one_hot[ann] = 1

        return img, one_hot

class COCOClassification(CocoDetection):
    def __init__(self, root, annFile, dataset_list=None, transform=None):
        super().__init__(root, annFile, transform)
        self.coco = COCO(annFile)
        self.coco_class = get_coco_class()
        self.coco_colormap = get_coco_colormap()
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

        if dataset_list is not None:
            with open(dataset_list, 'r') as f:
                img_id_strings = f.read().strip().split()
            
            # FIX: Convert string IDs to integers for proper matching
            target_img_ids = []
            for id_str in img_id_strings:
                try:
                    # Convert string ID to integer (removes leading zeros)
                    target_img_ids.append(int(id_str))
                except ValueError:
                    print(f"Warning: Could not convert '{id_str}' to integer, skipping")
                    continue
            
            # FIX: Use integer IDs directly instead of filename mapping
            all_img_ids = set(self.coco.getImgIds())
            self.ids = [img_id for img_id in target_img_ids if img_id in all_img_ids]
            
            print(f"COCOClassification: Loaded {len(self.ids)} images out of {len(target_img_ids)} requested")
            
            if len(self.ids) == 0:
                print("ERROR: No matching image IDs found!")
                print(f"Sample target IDs: {target_img_ids[:5]}")
                print(f"Sample available IDs: {list(all_img_ids)[:5]}")
        else:
            self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        label = torch.zeros(self.num_classes)
        for obj in target:
            cat_id = obj['category_id']
            label_idx = self.cat_id_to_idx[cat_id]
            label[label_idx] = 1
        return img, label
        
class COCO2014Segmentation(VisionDataset):
    def __init__(self, root, annFile, dataset_list=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.coco = COCO(annFile)
        self.root = root
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_idx = {cat_id: idx+1 for idx, cat_id in enumerate(self.cat_ids)}  # background = 0

        if dataset_list is not None:
            with open(dataset_list, 'r') as f:
                img_names = set(f.read().split())
            img_id_map = {v['file_name']: k for k, v in self.coco.imgs.items()}
            self.ids = [img_id_map[name] for name in img_names if name in img_id_map]
        else:
            self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        for obj in anns:
            cat_id = obj['category_id']
            cat_idx = self.cat_id_to_idx.get(cat_id, 0)
            seg_mask = self.coco.annToMask(obj)
            mask[seg_mask > 0] = cat_idx

        mask = Image.fromarray(mask.astype(np.uint8))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        return img, np.array(mask, dtype=np.uint8)

    def __len__(self):
        return len(self.ids)

def coco_train_dataset(args, img_list, mode='cls'):
    tfs_train = tfs.Compose([
        tfs.Resize((args.train['input_size'], args.train['input_size'])),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(args.train['crop_size'], padding=4, padding_mode='reflect'),
        tfs.ToTensor(),
        tfs.Normalize(voc_mean, voc_std),
    ])
    tfs_target = tfs.Compose([tfs.Resize((args.train['crop_size'], args.train['crop_size']))])

    if mode == 'cls':
        dataset = COCOClassification(
            root=os.path.join(args.dataset_root, 'train2014'),
            annFile=os.path.join(args.dataset_root, 'annotations/instances_train2014.json'),
            dataset_list=img_list,
            transform=tfs_train
        )
    elif mode == 'seg':
        dataset = COCO2014Segmentation(
            root=os.path.join(args.dataset_root, 'train2014'),
            annFile=os.path.join(args.dataset_root, 'annotations/instances_train2014.json'),
            dataset_list=img_list,
            transform=tfs_train,
            target_transform=tfs_target
        )
    return dataset

def coco_val_dataset(args, img_list, mode='cls'):
    tfs_val = tfs.Compose([
        tfs.Resize((args.eval['crop_size'], args.eval['crop_size'])),
        tfs.ToTensor(),
        tfs.Normalize(voc_mean, voc_std),
    ])
    tfs_target = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size']))])

    if mode == 'cls':
        dataset = COCOClassification(
            root=os.path.join(args.dataset_root, 'val2014'),
            annFile=os.path.join(args.dataset_root, 'annotations/instances_val2014.json'),
            dataset_list=img_list,
            transform=tfs_val
        )
    elif mode == 'seg':
        dataset = COCO2014Segmentation(
            root=os.path.join(args.dataset_root, 'val2014'),
            annFile=os.path.join(args.dataset_root, 'annotations/instances_val2014.json'),
            dataset_list=img_list,
            transform=tfs_val,
            target_transform=tfs_target
        )
    return dataset

