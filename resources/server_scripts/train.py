#------------------------------
# change able params
#------------------------------
PRETRAINED_WEIGHT_PATHS = "/home/nazmuddoha_ansary/work/apsisnetv2/model/rec_20_epochs_shortsteps.h5"

TRAIN_GCS_PATTERNS      = ["/backup2/apsisnetv2/tfrecords/*/*/*.tfrecord"]
                           
EVAL_GCS_PATTERNS       = ["/backup2/apsisnetv2/tfrecords/part_0/*/*.tfrecord"]

PER_REPLICA_BATCH_SIZE  = 64                          

EPOCHS                  = 2

#----------------
# imports
#---------------
import os
import warnings
import logging
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
import torch
import torch.nn as nn
#---------------------
# suppress warnings
#---------------------
# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all but error messages
# Suppress warnings globally
warnings.filterwarnings('ignore')
# Customize TensorFlow logger to show only errors
logging.getLogger('tensorflow').setLevel(logging.ERROR)


#---------------------
# GPU device setup
#---------------------

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        # Memory growth must be set before initializing GPUs
        print(e)
else:
    print("No GPU available.")


model_dir=os.path.join(os.getcwd(),"model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

#------------------------------------------------------------------------
# semi-fixed parameters
#------------------------------------------------------------------------
img_width =256
img_height=32
pos_max   =40
tf_size   =1024
vocab    = ["blank","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3",
            "4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G",
            "H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[",
            "\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o",
            "p","q","r","s","t","u","v","w","x","y","z","{","|","}","~","।","ঁ","ং","ঃ","অ",
            "আ","ই","ঈ","উ","ঊ","ঋ","এ","ঐ","ও","ঔ","ক","খ","গ","ঘ","ঙ","চ","ছ","জ","ঝ","ঞ",
            "ট","ঠ","ড","ঢ","ণ","ত","থ","দ","ধ","ন","প","ফ","ব","ভ","ম","য","র","ল","শ","ষ",
            "স","হ","া","ি","ী","ু","ূ","ৃ","ে","ৈ","ো","ৌ","্","ৎ","ড়","ঢ়","য়","০","১","২",
            "৩","৪","৫","৬","৭","৮","৯","‍","sep","pad"]
#-------------------
# fixed params
#------------------
nb_channels =  3    
enc_filters =  512

# calculated
pad_value   =  vocab.index("pad")
voc_len     =  len(vocab)

pos_emb              = nn.Embedding(pos_max+1,enc_filters)
pos_emb_weights      = pos_emb.weight.data.numpy()

print("Label len:",pos_max)
print("Vocab len:",voc_len)
print("Pad value:",pad_value)
print("pos embedding shape:",pos_emb_weights.shape)

#--------------------------
# GCS Paths and tfrecords
#-------------------------
train_recs=[]
eval_recs =[]
def get_tfrecs(gcs_pattern):
    file_paths = tf.io.gfile.glob(gcs_pattern)
    random.shuffle(file_paths)
    print(len(file_paths))
    return file_paths

for gcs in TRAIN_GCS_PATTERNS:
    print(gcs)
    train_recs+=get_tfrecs(gcs)
for gcs in EVAL_GCS_PATTERNS:
    print(gcs)
    eval_recs+=get_tfrecs(gcs)
# exclude evals
train_recs=[rec for rec in train_recs if rec not in eval_recs]
print("Eval-recs:",len(eval_recs))
print("Train-recs:",len(train_recs))
#----------------------------------------------------------
# Detect hardware, return appropriate distribution strategy
#----------------------------------------------------------
strategy = tf.distribute.get_strategy() 
# default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

#-------------------------------------
# batching , strategy and steps
#-------------------------------------
BATCH_SIZE = PER_REPLICA_BATCH_SIZE
# set    
STEPS_PER_EPOCH = (len(train_recs)*tf_size)//BATCH_SIZE
EVAL_STEPS      = (len(eval_recs)*tf_size)//BATCH_SIZE
print("Steps:",STEPS_PER_EPOCH)
print("Batch Size:",BATCH_SIZE)
print("Eval Steps:",EVAL_STEPS)

#------------------------------
# parsing tfrecords basic
#------------------------------
def data_input_fn(recs,mode): 
    '''
      This Function generates data from gcs
      * The parser function should look similiar now because of datasetEDA
        
        #         # mask
        #         mask=parsed_example['mask']
        #         mask=tf.image.decode_png(mask,channels=1)
        #         mask=tf.cast(mask,tf.float32)/255.0
        #         mask=tf.reshape(mask,(img_height,img_width,1))
        #         # lang
        #         lang=parsed_example['lang']
        #         lang = tf.strings.to_number(tf.strings.split(lang), out_type=tf.float32)
        #         lang = tf.reshape(lang,(1,))   

        #         return image,mask,std,label,lang
        #         return {"image":image,"mask":mask, "std": std, "lang": lang},label
        #         return image, label


    '''
    def _parser(example):   
        feature ={  'image' : tf.io.FixedLenFeature([],tf.string) ,
                    'mask' : tf.io.FixedLenFeature([],tf.string),
                    'std' : tf.io.FixedLenFeature([],tf.string),
                    'label':  tf.io.FixedLenFeature([],tf.string),
                    'lang':  tf.io.FixedLenFeature([],tf.string),
                  
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        # image
        image=parsed_example['std']
        image=tf.image.decode_png(image,channels=nb_channels)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(img_height,img_width,nb_channels))
        # label
        label=parsed_example['label']
        label = tf.strings.to_number(tf.strings.split(label), out_type=tf.float32)
        label = tf.reshape(label,(pos_max,))
        
        # position
        pos=tf.range(0,pos_max)
        pos=tf.cast(pos,tf.int32) 
        
        return {"image":image,"pos":pos},label

    
    # fixed code (for almost all tfrec training)
    dataset = tf.data.TFRecordDataset(recs)
    dataset = dataset.map(_parser,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1024,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset

# train ds
train_ds  =   data_input_fn(train_recs,"train")

# validation ds
eval_ds  =   data_input_fn(eval_recs,"eval")

#------------------------
# visualizing data
#------------------------
langs=["bn","en"]
print("---------------------------------------------------------------")
print("visualizing data")
print("---------------------------------------------------------------")
for data,label in train_ds.take(1):
    images=data["image"]
    posis=data["pos"]
    print("image")
    data=np.squeeze(images[0])
    plt.imshow(data)
    plt.show()    
    print("---------------------------------------------------------------")
    _label=label[0].numpy()
    print(_label)
    text="".join([vocab[int(c)] for c in _label if vocab[int(c)] not in ["pad","sep"]])
    print("label :",text)
    print("---------------------------------------------------------------")
    print('Batch Shape:',images.shape)
    print("---------------------------------------------------------------")

#     print("Positional encoding:",posis[0])
#     print("mask")
#     data=np.squeeze(mask[0])
#     plt.imshow(data)
#     plt.show()
#     print("std")
#     data=np.squeeze(std[0])
#     plt.imshow(data)
#     plt.show()
#     print("---------------------------------------------------------------")
#     _lang=lang[0].numpy()
#     _lang=langs[int(_lang)]
#     print("lang:",_lang)


#--------------------------------------------
# custom layers
#--------------------------------------------

class DotAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.inf_val=-1e9
        
    def call(self,q, k, v, mask):
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
       
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * self.inf_val)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output
     
    def get_config(self):
        config = super().get_config().copy()
        return config
        
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,num_seq,projection_dim):
        super(PositionalEncoding, self).__init__()
        self.projection_dim=projection_dim
        self.num_seq = num_seq
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_seq, output_dim=projection_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_seq, delta=1)
        encoded = self.projection(x) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_seq': self.num_seq,
                       'projection_dim':self.projection_dim})
        return config

    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.ff_dim   =ff_dim

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self,x,mask,training):
        attn_output = self.att(query=x,key=x,value=x,attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim,
                       'num_heads': self.num_heads,
                       'ff_dim':self.ff_dim})
        return config
    


#--------------------------------------------
# custom blocks 
#--------------------------------------------

def ConvBlock(x,filters):
    x=tf.keras.layers.Conv2D(filters,3,padding='same')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x=tf.keras.layers.Activation("relu")(x)
    return x

def attend(x,mask,num_heads,num_blocks,reshape=True):
    bs,h,w,nc=x.shape
    x = tf.keras.layers.Reshape((h*w,nc))(x)
    x = PositionalEncoding(h*w,nc)(x)
    for _ in range(num_blocks):
        x=TransformerBlock(embed_dim=nc, num_heads=num_heads,ff_dim=4*nc)(x,mask)
    if reshape:
        x = tf.keras.layers.Reshape((h,w,nc))(x)
    return x

#--------------------------------------------------------
# metrics and losses
#--------------------------------------------------------

def C_acc(y_true, y_pred):
    accuracies = tf.equal(tf.cast(y_true,tf.int64), tf.argmax(y_pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(y_true,pad_value))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class CharLoss(tf.keras.losses.Loss):
    def __init__(self,pad_value):
        super(CharLoss, self).__init__(name="char_loss")
        self.pad_value=pad_value
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def call(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, pad_value))
        loss_ = self.loss_object(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self,pad_value,logits_time_major=False,name='ctc'):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.pad_value=pad_value

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        label_mask = tf.cast(y_true!= self.pad_value, tf.int32)
        label_length = tf.reduce_sum(label_mask, axis=-1)
        
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.pad_value)
        return tf.reduce_mean(loss)

#------------------------------------------------------------------
# callbacks 
#------------------------------------------------------------------

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(patience=40, 
                                                  verbose=1, 
                                                  mode = 'auto')

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self,model_dir):
        self.best = float('inf')
        self.output_dir = model_dir

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs['val_loss']
        if metric_value < self.best:
            print(f"Loss Improved epoch:{epoch} from {self.best} to {metric_value}",end="#")
            self.best = metric_value
            save_path = os.path.join(self.output_dir, "rec_best.h5")
            self.model.save_weights(save_path)
            print("Saved Weights")
    def set_model(self, model):
        self.model = model

def build_model():
    img_shape=(img_height,img_width,nb_channels)
    img =tf.keras.Input(shape=img_shape,name="image")
    pos =tf.keras.Input(shape=(pos_max,),name="pos")
    x=ConvBlock(img,64)
    x=ConvBlock(x,128)
    x=attend(x,None,4,4)
    x=ConvBlock(x,256)
    x=ConvBlock(x,512)
    x=attend(x,None,8,8,reshape=False)
    #------------pos encoding------------------
    query=tf.keras.layers.Embedding(input_dim=pos_max+1, output_dim=enc_filters,weights=[pos_emb_weights])(pos)
    attn=DotAttention()(query,x,x,None)
    x=tf.keras.layers.Dense(voc_len,activation=None,name="logits")(attn)
    model = tf.keras.Model(inputs=[img,pos],outputs=x)
    return model

with strategy.scope():
    lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=0.0001,decay_steps=600000,alpha= 0.01)
    model = build_model()
    if PRETRAINED_WEIGHT_PATHS is not None:
        model.load_weights(PRETRAINED_WEIGHT_PATHS)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                  loss=CharLoss(pad_value),
                  metrics=[C_acc])

# call back setup
model_save=SaveBestModel(model_dir)
model_save.set_model(model)
callbacks = [model_save,early_stopping]


history=model.fit(train_ds,
                  epochs=EPOCHS,
                  steps_per_epoch=STEPS_PER_EPOCH,
                  verbose=1,
                  validation_data=eval_ds,
                  validation_steps=EVAL_STEPS, 
                  callbacks=callbacks)

curves={}
for key in history.history.keys():
    curves[key]=history.history[key]
curves=pd.DataFrame(curves)
curves.to_csv(f"history.csv",index=False)