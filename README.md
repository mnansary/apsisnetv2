# apsisnetv2
apsisnet version 2.0

# Environment Setup

## Conda Environment

```bash
conda create -n apsisnetv2 python=3.9
conda activate apsisnetv2
bash install.sh
```

## Server Information

```bash
            .-/+oossssoo+/-.               NEOFETCH INFO 
        `:+ssssssssssssssssss+:`           ------------------------ 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 22.04.4 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: Z490 GAMING X AX -CF 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 6.5.0-18-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/     
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: bash 5.1.16 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Terminal: node 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   CPU: Intel i9-10900K (20) @ 5.300GHz 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   GPU: NVIDIA GeForce RTX 3090 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Memory: 32006MiB 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/                            
  +sssssssssdmydMMMMMMMMddddyssssssss+                             
   /ssssssssssshdmNNNNmyNMMMMhssssss/
    .ossssssssssssssssssdMMMNysssso.
      -+sssssssssssssssssyyyssss+-
        `:+ssssssssssssssssss+:`
            .-/+oossssoo+/-.
```
```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.06             Driver Version: 535.183.06   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        On  | 00000000:01:00.0 Off |                  N/A |
|  0%   41C    P8              19W / 370W |     26MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

# Data Preparation

The following sources are used for data preparation:

| Source Type|Data Level|Description|language|source|link|identifier | datacount |
|------------|----------|-----------|--------|------|----|-----------|-----------|
| Synthetic|Word| |English|synthtiger| https://github.com/clovaai/synthtiger |synthtiger-english|10000000|
| Synthetic|Word| | Bangla|synthtiger| https://github.com/clovaai/synthtiger |synthtiger-bangla|14212387|
| Synthetic|Word| |Bangla|synthindic| https://github.com/BengaliAI/synthIndic |synthindic-bangla|2139971|
|handwriten|grapheme|handwriten grapheme dataset - kaggle competition|Bangla|Bengali.ai| https://www.kaggle.com/competitions/bengaliai-cv19 |isolated-bangla-grapheme|200840|
|handwriten|numbers|handwriten numbers dataset - numta kaggle competition |Bangla|Bengali.ai| https://www.kaggle.com/competitions/numta |isolated-bangla-numta|72045|
|handwriten|word|handwriten word dataset created from a mixure of datasets</br> (1)Bangla Writing </br> (2)Boise State Dataset </br> (3)BN-HTRd </br> (4)IIIT BN data </br> **Note: IIIT-BN is completely unused for training and only used for testing** |Bangla|Online| (1)https://paperswithcode.com/dataset/banglawriting </br> (2)https://scholarworks.boisestate.edu/saipl/1/ </br> (3)https://arxiv.org/abs/2206.08977 </br> (4)https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words |natural-handwriten-bangla|187811|
|scene|word|natural scene dataset created from a mixure of datasets</br>(1)MLT2017</br>(2)MLT2019</br> **Note**: Only taken the word crops that has bangla labels and has no space in label|Bangla|Online| (1)https://rrc.cvc.uab.es/?ch=8 </br> (2) https://rrc.cvc.uab.es/?ch=15 |natural-scene-bangla|7885 |
|handwriten|word|handwriten word dataset created from a mixure of datasets</br> (1)IAM English Data </br>  (2)GNHK Dataset |English|Online| (1)https://paperswithcode.com/dataset/iam </br> (2)https://assets.amazon.science/38/fe/4c3105fb43129bf59cc0aadb5d78/gnhk-a-dataset-for-english-handwriting-in-the-wild.pdf |natural-handwriten-english|157446 |
|mixed|word|Scene,Printed,Memo etc. word dataset created from a mixure of datasets</br> (1)SROIE </br> (2)ICDAR 2015 </br> (3)MLT2017 </br> (4)MLT2019 </br> (5)Wild receipt</br> (6) FUNSD </br> (7)MSRA-TD500 </br> (8)CTW-1500 </br> **Note**: Only taken the word crops that has English labels and has no space in label|English|Online| (1)https://paperswithcode.com/dataset/sroie </br> (2)https://rrc.cvc.uab.es/?ch=4 </br> (3)https://rrc.cvc.uab.es/?ch=8 </br> (4) https://rrc.cvc.uab.es/?ch=15 </br> (5)https://paperswithcode.com/dataset/wildreceipt </br> (6)https://paperswithcode.com/dataset/funsd </br> (7) https://paperswithcode.com/dataset/msra-td500 </br> (8)https://www.kaggle.com/datasets/ipythonx/ctw1500-str  |natural-mixed-english|188629 |

## Processed Dataset
The processed tfrecord datasets are listed in kaggle as follows: 
1) https://www.kaggle.com/datasets/ocrteamriad/apsisnetv2-data-p0-4 
2) https://www.kaggle.com/datasets/ocrteamriad/apsisnetv2-data-p5-9 
3) https://www.kaggle.com/datasets/ocrteamriad/apsisnetv2-data-p10-14 
4) https://www.kaggle.com/datasets/ocrteamriad/apsisnetv2-data-p15-19 
5) https://www.kaggle.com/datasets/ocrteamriad/apsisnetv2-data-p20-24
6) https://www.kaggle.com/datasets/ocrteamriad/apsisnetv2-data-p25-27 

## Data Generation
* useage: ```python data_gen.py```

```python
usage: ApsisNetv2 tfrecord Dataset Creation Script [-h] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] [--label_max_len LABEL_MAX_LEN] [--num_process NUM_PROCESS]
                                                   data_csv vocab_txt save_path

positional arguments:
  data_csv              Path of the data_csv holding absolute image path,word and language [cols:filepath,word,lang]
  vocab_txt             Path of the vocab.txt file holding the unicodes to use
  save_path             Path of the directory to save the tfrecord dataset

optional arguments:
  -h, --help            show this help message and exit
  --img_height IMG_HEIGHT
                        the desired height of the image:default=32
  --img_width IMG_WIDTH
                        the desired width of the image:default=256
  --label_max_len LABEL_MAX_LEN
                        maximum length for the text label:default=40
  --num_process NUM_PROCESS
                        number of processes to be used:default=16

```

# Recognizer Trainig 

* For training ```notebooks/train_recognizer.ipynb``` is used. 
* Check notebook first cell for variable parameters

### Stage-1 
* lowering the Training steps and training for 20 epochs 

```python
PRETRAINED_WEIGHT_PATHS = None
...
...
...
STEPS_PER_EPOCH = ((len(train_recs)*tf_size)//(BATCH_SIZE))//10
EVAL_STEPS      = ((len(eval_recs)*tf_size)//(BATCH_SIZE))//5
```

* results

|loss|C_acc|val_loss|val_C_acc|
|----|-----|--------|---------|
|0.0019560614600777626|0.9994895458221436|0.0015700346557423472|0.999584972858429|

* time= 7100s (+-10s) per epoch
 
### Stage-2 
* Original Training steps and training for 2 epochs from the previous step 