#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import json
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from datalib.store import createRecords
from datalib.utils import LOG_INFO, create_dir
from datalib.constants import SPLIT
from multiprocessing import Process
#------------------------
# data
#-------------------------
def get_vocab(vocab_txt):
    vocab=[]
    with open(vocab_txt,"r") as f:
        lines=f.readlines()
    for line in lines:
        if line.strip():
            vocab.append(line.strip())
    vocab=["blank"]+vocab+["sep","pad"]
    return vocab

def check_text(x,vocab):
    for i in x:
        if i not in vocab:
            return None
    return x


def main(args):
    data_csv    =   args.data_csv
    vocab_txt   =   args.vocab_txt
    save_path   =   args.save_path
    num_proc    =   int(args.num_process)
    
    
    save_path   =   create_dir(save_path,"tfrecords")
    temp_path   =   create_dir(save_path,"temp")
    
    class cfg:
        img_height = int(args.img_height)
        img_width  = int(args.img_width)
        pos_max    = int(args.label_max_len)
        vocab      = get_vocab(vocab_txt)

    df=pd.read_csv(data_csv)
    print("total data:",len(df))
    df["word"]=df["word"].progress_apply(lambda x:str(x))
    df["word"]=df["word"].progress_apply(lambda x: x if len(x)<cfg.pos_max-2 else None)
    df.dropna(inplace=True)
    df["word"]=df["word"].progress_apply(lambda x: check_text(x,cfg.vocab))
    df.dropna(inplace=True)
    print("filtered number of data:",len(df))
    dfs=[df[idx:idx+SPLIT] for idx in range(0,len(df),SPLIT)]
    max_end=len(dfs)
    

    def run(idx):
        if idx <len(dfs):
            tf_path=create_dir(save_path,str(idx))
            createRecords(dfs[idx],tf_path,idx,temp_path,cfg)


    def execute(start,end):
        process_list=[]
        for idx in range(start,end):
            p =  Process(target= run, args = [idx])
            p.start()
            process_list.append(p)
        for process in process_list:
            process.join()


    if max_end==1:
        dfs=[df]
        run(0)
    elif max_end<=num_proc:
        for i in range(0,max_end):
            start=i
            end=start+max_end-1
            execute(start,end) 
    else:
        for i in range(0,max_end,num_proc):
            start=i
            end=start+num_proc
            if end>max_end:end=max_end-1
            execute(start,end) 

    

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("ApsisNetv2 tfrecord Dataset Creation Script")
    parser.add_argument("data_csv", help="Path of the data_csv holding absolute image path,word and language [cols:filepath,word,lang]")
    parser.add_argument("vocab_txt", help="Path of the vocab.txt file holding the unicodes to use")
    parser.add_argument("save_path", help="Path of the directory to save the tfrecord dataset")
    
    parser.add_argument("--img_height",required=False,default=32,help ="the desired height of the image:default=32")
    parser.add_argument("--img_width",required=False,default=256,help ="the desired width of the image:default=256")
    parser.add_argument("--label_max_len",required=False,default=40,help ="maximum length for the text label:default=40")
    
    parser.add_argument("--num_process",required=False,default=16,help ="number of processes to be used:default=16")
    
    args = parser.parse_args()
    main(args)
