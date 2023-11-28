# apsisnetv2
apsisnet version 2.0

# Data Gen
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