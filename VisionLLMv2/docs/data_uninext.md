# UNINEXT Model and Data Preparation

Make sure you are now on the `./VisionLLMv2` folder.


# Model

```
mkdir checkpoints && cd checkpoints
ln -s /mnt/petrelfs/share_data/liuzhaoyang/share/wjn_models/bert-base-uncased .
ln -s /mnt/petrelfs/share_data/liuzhaoyang/share/wjn_models/uninext .
cd ..
```

```
mkdir work_dirs && cd work_dirs
ln -s /mnt/petrelfs/share_data/liuzhaoyang/share/wjn_models/visionllmv2-7b-pure .
ln -s /mnt/petrelfs/share_data/liuzhaoyang/share/wjn_models/visionllmv2-7b-pure-ft-uninext .
ln -s /mnt/petrelfs/share_data/liuzhaoyang/share/wjn_models/visionllmv2-internvl2-8b-ft-uninext .
cd ..
```



# Data

```
mkdir datasets && cd datasets
```

## 1 OD & IS

### COCO

```
ln -s /mnt/petrelfs/share_data/chenzhe1/data/coco .
```

## 2 REC & RES

```
ln -s /mnt/petrelfs/share_data/wujiannan/data/coco2014/annotations .
```

## 3 VIS

### YTVIS2019

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/ytvis2019/ ytvis_2019
```

### OVIS

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/OVIS ovis
```

## 4 VOS

### YTVOS2018

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/youtube-vos18 ytbvos18
```

### DAVIS

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/DAVIS/ .
```


## 5 RVOS

### RefYTVOS

```
ln -s mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/ref-youtube-vos .
```

### RefDAVIS

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/ref-davis .
```


## 6 SOT

### LaSOT

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/LaSOTTest/LaSOTTest LaSOT
```

### LaSOText

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/LaSOT_extension_subset .
```

### TrackingNet

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/TrackingNet_HF TrackingNet
```

### TNL2K

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/TNL2K_test_subset/TNL-2K .
```


## MOT & MOTS


### 7 BDD100K

```
ln -s /mnt/petrelfs/share_data/liuzhaoyang/datasets/wjn_data/bdd100k/bdd100k bdd
```


# Demo

Refer to `demo/demo_vllmv2-7b.sh` for VisionLLMv2-7B + UNINEXT demo.
Refer to `demo/demo_vllmv2-internvl2-8b.sh` for VisionLLMv2-InternVL2-8B + UNINEXT demo. (Recommend)