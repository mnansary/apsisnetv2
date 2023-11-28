# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np 
from tqdm.auto import tqdm
import cv2
from .utils import correctPadding,createTextImage
from .constants import LANGS
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def toTfrecord(df,rnum,rec_path,sidx,temp_path,cfg):
    tfrecord_name=f'{sidx}_{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            try:
                img_path=df.iloc[idx,0]
                word=df.iloc[idx,1]
                lang=df.iloc[idx,2]

                img=cv2.imread(img_path)
                img,mask=correctPadding(img,(cfg.img_height,cfg.img_width))
                std=createTextImage(word,lang)
                std,_=correctPadding(std,(cfg.img_height,cfg.img_width))
                
                tmp_img_path=os.path.join(temp_path,f"{sidx}_img.png")
                tmp_msk_path=os.path.join(temp_path,f"{sidx}_msk.png")
                tmp_std_path=os.path.join(temp_path,f"{sidx}_std.png")
                
                cv2.imwrite(tmp_img_path,img)
                cv2.imwrite(tmp_msk_path,mask)
                cv2.imwrite(tmp_std_path,std)
                
                # img
                with(open(tmp_img_path,'rb')) as fid:
                    image_png_bytes=fid.read()
                # msk
                with(open(tmp_msk_path,'rb')) as fid:
                    msk_png_bytes=fid.read()
                
                # std
                with(open(tmp_std_path,'rb')) as fid:
                    std_png_bytes=fid.read()
                
                # word
                word=[cfg.vocab.index(c) for c in word]
                word=[cfg.vocab.index('sep')]+word+[cfg.vocab.index('sep')]
                for _ in range(cfg.pos_max-len(word)):
                    word+=[cfg.vocab.index('pad')]
                label=" ".join([str(x) for x in word])
                label=bytes(label, "utf-8")
                
                # lang
                lang=LANGS.index(lang)
                lang=bytes(str(lang),"utf-8")

                
                # feature desc
                data ={ 'image':_bytes_feature(image_png_bytes)}
                data["mask"]=_bytes_feature(msk_png_bytes)
                data["std"]=_bytes_feature(std_png_bytes)               
                data["label"]=_bytes_feature(label)
                data["lang"]=_bytes_feature(lang)
                
                
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)  
            except Exception as e:
                print(img_path)
                print(word)
                print(e)
                
def createRecords(data,save_path,sidx,temp_path,cfg,tf_size=1024):
    print(f"Creating TFRECORDS:{save_path}")
    for idx in tqdm(range(0,len(data),tf_size)):
        df        =   data.iloc[idx:idx+tf_size] 
        df.reset_index(drop=True,inplace=True) 
        rnum      =   idx//tf_size
        toTfrecord(df,rnum,save_path,sidx,temp_path,cfg)

    
    