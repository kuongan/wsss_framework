#!/bin/bash

# === Set thư mục lưu dữ liệu ===
DATA_DIR="./coco"
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "📥 Downloading COCO 2014 train images..."
wget -c http://images.cocodataset.org/zips/train2014.zip

echo "📥 Downloading COCO 2014 val images..."
wget -c http://images.cocodataset.org/zips/val2014.zip

echo "📥 Downloading COCO 2014 annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

echo "📦 Unzipping train2014.zip..."
unzip -q train2014.zip

echo "📦 Unzipping val2014.zip..."
unzip -q val2014.zip

echo "📦 Unzipping annotations..."
unzip -q annotations_trainval2014.zip

echo "🧹 Cleaning zip files..."
rm train2014.zip val2014.zip annotations_trainval2014.zip

echo "✅ COCO 2014 dataset ready!"
