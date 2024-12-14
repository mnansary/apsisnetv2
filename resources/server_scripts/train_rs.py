#------------------------------
# change able params
#------------------------------
PRETRAINED_WEIGHT_DIR   = "/home/nazmuddoha_ansary/work/apsisnetv2/model/robust_scanner/"

TRAIN_GCS_PATTERNS      = ["/home/nazmuddoha_ansary/work/apsisnetv2/tfrecords/*/*/*.tfrecord"]
                           
EVAL_GCS_PATTERNS       = ["/home/nazmuddoha_ansary/work/apsisnetv2/tfrecords/part_0/*/*.tfrecord"]

PER_REPLICA_BATCH_SIZE  = 512                          

EPOCHS                  = 20

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
enc_filters =  256
factor      =  32

# calculated
enc_shape   =  (img_height//factor,img_width//factor, enc_filters )
attn_shape  =  (None, enc_filters )
mask_len    =  int((img_width//factor)*(img_height//factor))

sep_value   =  vocab.index("sep")
pad_value   =  vocab.index("pad")
voc_len     =  len(vocab)

print("Label len:",pos_max)
print("Vocab len:",voc_len)
print("Pad value:",pad_value)
print("Sep value:",sep_value)

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
DECAY_STEPS     = EPOCHS * (STEPS_PER_EPOCH+EVAL_STEPS) 
print("Steps:",STEPS_PER_EPOCH)
print("Batch Size:",BATCH_SIZE)
print("Eval Steps:",EVAL_STEPS)
print("Decay Steps:",DECAY_STEPS)

#------------------------------
# parsing tfrecords basic
#------------------------------

def data_input_fn(recs,mode): 
    '''
      This Function generates data from gcs
      * The parser function should look similiar now because of datasetEDA
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
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=nb_channels)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(img_height,img_width,nb_channels))
        # label
        label=parsed_example['label']
        label = tf.strings.to_number(tf.strings.split(label), out_type=tf.float32)
        # position
        pos=tf.range(0,pos_max)
        pos=tf.cast(pos,tf.int32)
        # mask
        mask=parsed_example['mask']
        mask=tf.image.decode_png(mask,channels=1)
        mask=tf.cast(mask,tf.float32)/255.0
        mask=tf.reshape(mask,(img_height,img_width,1))
        mask=tf.image.resize(mask,[img_height//factor,img_width//factor],method="nearest")
        mask=tf.reshape(mask,[-1])
        mask=tf.stack([mask for _ in range(pos_max)])
        return {"image":image,"label":tf.cast(label, tf.int32),"pos":pos,"mask":mask},label


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


print("---------------------------------------------------------------")
print("visualizing data")
print("---------------------------------------------------------------")
for x,y in train_ds.take(1):
    data=np.squeeze(x["image"][0])
    plt.imshow(data)
    plt.show()
    print("---------------------------------------------------------------")
    print("label:",x["label"][0])
    _label=x['label'][0].numpy()
    text="".join([vocab[int(c)] for c in _label if vocab[int(c)] not in ["pad","sep"]])
    print("label-text :",text)
    print("---------------------------------------------------------------")
    print("pos:",x["pos"][0])
    print("---------------------------------------------------------------")
    print("mask:",x["mask"][0][0])
    print("---------------------------------------------------------------")
    print('Image Batch Shape:',x["image"].shape)
    print('Label Batch Shape:',x["label"].shape)
    print('Position Batch Shape:',x["pos"].shape)
    print('Mask Batch Shape:',x["mask"].shape)
    print("---------------------------------------------------------------")
    print('Target Batch Shape:',y.shape)

#------------------------------
# pretrained weights paths
#------------------------------
enc_weights             = os.path.join(PRETRAINED_WEIGHT_DIR ,"enc.h5")
seq_weights             = os.path.join(PRETRAINED_WEIGHT_DIR ,"seq.h5")
pos_weights             = os.path.join(PRETRAINED_WEIGHT_DIR ,"pos.h5")
fuse_weights            = os.path.join(PRETRAINED_WEIGHT_DIR ,"fuse.h5")

#-----------------------------------
#creating Embedding Weights
#-----------------------------------
seq_emb              = nn.Embedding(voc_len+1,enc_filters, padding_idx=pad_value)
seq_emb_weight       = seq_emb.weight.data.numpy()
pos_emb              = nn.Embedding(pos_max+1,enc_filters)
pos_emb_weight       = pos_emb.weight.data.numpy()

#---------------------------------------------------
# dot attention layer
#---------------------------------------------------
class DotAttention(tf.keras.layers.Layer):
    """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output
    """
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
    
def encoder():
    '''
    creates the encoder part:
    * defatult backbone : DenseNet121 **changeable
    args:
      img           : input image layer
        
    returns:
      enc           : channel reduced feature layer

    '''
    # img input
    img=tf.keras.Input(shape=(img_height,img_width,nb_channels),name='image')
    
    cnn = tf.keras.applications.ResNet50V2(input_tensor=img,weights=None,include_top=False)
    enc = cnn.output
    # enc 
    enc=tf.keras.layers.Conv2D(enc_filters,kernel_size=3,padding="same")(enc)

    return tf.keras.Model(inputs=img,outputs=enc,name="rs_encoder")

def seq_decoder():
    '''
    sequence attention decoder (for training)
    Tensorflow implementation of : 
    https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/sequence_attention_decoder.py
    '''
    # label input
    gt=tf.keras.Input(shape=(pos_max,),dtype='int32',name="label")
    # mask
    mask=tf.keras.Input(shape=(pos_max,mask_len),dtype='float32',name="mask")
    # encoder
    enc=tf.keras.Input(shape=enc_shape,name='enc_seq')
    # embedding
    embedding=tf.keras.layers.Embedding(voc_len+1,enc_filters,weights=[seq_emb_weight])(gt)
    # sequence layer (2xlstm)
    lstm=tf.keras.layers.LSTM(enc_filters,return_sequences=True)(embedding)
    query=tf.keras.layers.LSTM(enc_filters,return_sequences=True)(lstm)
    # attention modeling
    # value
    bs,h,w,nc=enc.shape
    value=tf.keras.layers.Reshape((h*w,nc))(enc)
    attn=DotAttention()(query,value,value,mask)
    return tf.keras.Model(inputs=[gt,enc,mask],outputs=attn,name="rs_seq_decoder")
 


def pos_decoder():
    '''
    position attention decoder (for training)
    Tensorflow implementation of : 
    https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/position_attention_decoder.py
    '''
    # pos input
    pt=tf.keras.Input(shape=(pos_max,),dtype='int32',name="pos")
    # mask
    mask=tf.keras.Input(shape=(pos_max,mask_len),dtype='float32',name="mask")
    # encoder
    enc=tf.keras.Input(shape=enc_shape,name='enc_pos')
    
    # embedding,weights=[pos_emb_weight]
    query=tf.keras.layers.Embedding(pos_max+1,enc_filters,weights=[pos_emb_weight])(pt)
    # part-1:position_aware_module
    bs,h,w,nc=enc.shape
    value=tf.keras.layers.Reshape((h*w,nc))(enc)
    # sequence layer (2xlstm)
    lstm=tf.keras.layers.LSTM(enc_filters,return_sequences=True)(value)
    x=tf.keras.layers.LSTM(enc_filters,return_sequences=True)(lstm)
    x=tf.keras.layers.Reshape((h,w,nc))(x)
    # mixer
    x=tf.keras.layers.Conv2D(enc_filters,kernel_size=3,padding="same")(x)
    x=tf.keras.layers.Activation("relu")(x)
    key=tf.keras.layers.Conv2D(enc_filters,kernel_size=3,padding="same")(x)
    bs,h,w,c=key.shape
    key=tf.keras.layers.Reshape((h*w,nc))(key)
    attn=DotAttention()(query,key,value,mask)
    return tf.keras.Model(inputs=[pt,enc,mask],outputs=attn,name="rs_pos_decoder")

def fusion():
    '''
    fuse the output of gt_attn and pt_attn 
    '''
    # label input
    gt_attn=tf.keras.Input(shape=attn_shape,name="gt_attn")
    # pos input
    pt_attn=tf.keras.Input(shape=attn_shape,name="pt_attn")
    
    x=tf.keras.layers.Concatenate()([gt_attn,pt_attn])
    # Linear
    x=tf.keras.layers.Dense(enc_filters*2,activation=None)(x)
    # GLU
    xl=tf.keras.layers.Activation("linear")(x)
    xs=tf.keras.layers.Activation("sigmoid")(x)
    x =tf.keras.layers.Multiply()([xl,xs])
    # prediction
    x=tf.keras.layers.Dense(voc_len,activation=None)(x)
    return tf.keras.Model(inputs=[gt_attn,pt_attn],outputs=x,name="rs_fusion")

with strategy.scope():
    rs_encoder    =  encoder()
    rs_seq_decoder=  seq_decoder()
    rs_pos_decoder=  pos_decoder()
    rs_fusion     =  fusion()
    if os.path.exists(enc_weights):
        rs_encoder.load_weights(enc_weights)
        print("enc:",enc_weights)
    if os.path.exists(seq_weights):
        rs_seq_decoder.load_weights(seq_weights)
        print("seq:",seq_weights)
    if os.path.exists(pos_weights):
        rs_pos_decoder.load_weights(pos_weights)
        print("pos:",pos_weights)
    if os.path.exists(fuse_weights):
        rs_fusion.load_weights(fuse_weights)
        print("fuse:",fuse_weights)

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.0001,
                                                 decay_steps=DECAY_STEPS,
                                                 alpha= 0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def CE_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, pad_value))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def C_acc(real, pred):
    accuracies = tf.equal(tf.cast(real,tf.int64), tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real,pad_value))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


class robust_scanner(tf.keras.Model):
    def __init__(self,encoder,seq_decoder,pos_decoder,fusion):
        super(robust_scanner, self).__init__()
        self.encoder     = encoder
        self.seq_decoder = seq_decoder
        self.pos_decoder = pos_decoder
        self.fusion      = fusion

    def compile(self,optimizer,loss_fn,acc):
        super(robust_scanner, self).compile(optimizer=optimizer)
        self.optimizer = optimizer
        self.loss_fn   = loss_fn
        self.acc       = acc


    def train_step(self, batch_data):
        data,gt= batch_data
        image=data["image"]
        pos  =data["pos"]
        mask =data["mask"]
        
        with tf.GradientTape() as enc_tape, tf.GradientTape() as pos_dec_tape,tf.GradientTape() as seq_dec_tape,tf.GradientTape() as fusion_tape:
            enc    = self.encoder(image)
            pt_attn= self.pos_decoder([pos,enc,mask])

            gt_attn= self.seq_decoder([gt,enc,mask])
            pred   = self.fusion([gt_attn,pt_attn])

            # loss
            loss = self.loss_fn(gt[:,1:],pred[:,:-1,:])
            # c acc
            char_acc=self.acc(gt[:,1:],pred[:,:-1,:])

        # calc gradients    
        enc_grads     = enc_tape.gradient(loss,self.encoder.trainable_variables)
        pos_dec_grads = pos_dec_tape.gradient(loss,self.pos_decoder.trainable_variables)
        seq_dec_grads = seq_dec_tape.gradient(loss,self.seq_decoder.trainable_variables)
        fusion_grads  = fusion_tape.gradient(loss,self.fusion.trainable_variables)

        # apply
        self.optimizer.apply_gradients(zip(enc_grads,self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(pos_dec_grads,self.pos_decoder.trainable_variables))

        self.optimizer.apply_gradients(zip(seq_dec_grads,self.seq_decoder.trainable_variables))
        self.optimizer.apply_gradients(zip(fusion_grads,self.fusion.trainable_variables))


        return {"loss"    : loss,
                "char_acc": char_acc}

    def test_step(self, batch_data):
        data,gt= batch_data
        image=data["image"]
        pos  =data["pos"]
        mask =data["mask"]
        # label
        label=tf.ones_like(gt,dtype=tf.float32)*sep_value
        preds=[]

        enc    = self.encoder(image, training=False)
        pt_attn= self.pos_decoder([pos,enc,mask],training=False)

        for i in range(pos_max):
            gt_attn=self.seq_decoder([label,enc,mask],training=False)
            step_gt_attn=gt_attn[:,i,:]
            step_pt_attn=pt_attn[:,i,:]
            pred=self.fusion([step_gt_attn,step_pt_attn],training=False)
            preds.append(pred)
            # can change on error
            char_out=tf.nn.softmax(pred,axis=-1)
            max_idx =tf.math.argmax(char_out,axis=-1)
            if i < pos_max - 1:
                label=tf.unstack(label,axis=-1)
                label[i+1]=tf.cast(max_idx,tf.float32)
                label=tf.stack(label,axis=-1)

        pred=tf.stack(preds,axis=1)
        # loss
        loss = self.loss_fn(gt[:,1:],pred[:,:-1,:])
        # c acc
        char_acc=self.acc(gt[:,1:],pred[:,:-1,:])


        return {"loss"    : loss,
                "char_acc": char_acc}
    
with strategy.scope():
    model = robust_scanner(rs_encoder,
                           rs_seq_decoder,
                           rs_pos_decoder,
                           rs_fusion)

model.compile(optimizer = optimizer,
              loss_fn   = CE_loss,
              acc       = C_acc)
    
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs['val_loss']
        if metric_value < self.best:
            print(f"Loss Improved epoch:{epoch} from {self.best} to {metric_value}")
            self.best = metric_value
            self.model_to_save.encoder.save_weights(enc_weights)
            self.model_to_save.seq_decoder.save_weights(seq_weights)
            self.model_to_save.pos_decoder.save_weights(pos_weights)
            self.model_to_save.fusion.save_weights(fuse_weights)
            print("Saved Weights")
    def set_model(self, model):
        self.model_to_save = model  # Assign the model to a different attribute

            
model_save=SaveBestModel()
model_save.set_model(model)
callbacks= [model_save]

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
curves.to_csv(f"history_robust_scanner_20_epochs.csv",index=False)