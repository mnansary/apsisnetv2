#------------------------------
# change able params
#------------------------------
RECOGNIZER_WEIGHT_PATH = "/home/nazmuddoha_ansary/work/apsisnetv2/model/rec_binarized.h5"

TRAIN_GCS_PATTERNS      = ["/home/nazmuddoha_ansary/work/apsisnetv2/tfrecords/*/*/*.tfrecord"]
                           
EVAL_GCS_PATTERNS       = ["/home/nazmuddoha_ansary/work/apsisnetv2/tfrecords/part_0/*/*.tfrecord"]

GENERATOR_WEIGHT_PATH    = "/home/nazmuddoha_ansary/work/apsisnetv2/model/generator_best.h5"

DISCRIMINATOR_WEIGHT_PATH ="/home/nazmuddoha_ansary/work/apsisnetv2/model/discriminator_best.h5" 


PER_REPLICA_BATCH_SIZE  = 64                         

EPOCHS                  = 10

GENERATOR_BACKBONE      = 'densenet121'
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

#-----------------------------------------
# segmentation model backend setup
#-----------------------------------------
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
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
vocab    = ["\u200d","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3",
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
sep_value   =  vocab.index("sep") 
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
DECAY_STEPS     = EPOCHS * (STEPS_PER_EPOCH+EVAL_STEPS) 
print("Steps:",STEPS_PER_EPOCH)
print("Batch Size:",BATCH_SIZE)
print("Eval Steps:",EVAL_STEPS)
print("Decay Steps:",DECAY_STEPS)

#------------------------------
# parsing tfrecords basic
#------------------------------
def data_input_fn(recs,mode,threshold=0.5): 
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
        image=parsed_example['image']
        image=tf.image.decode_png(image,channels=nb_channels)
        image=(tf.cast(image,tf.float32)/127.5) -1
        image=tf.reshape(image,(img_height,img_width,nb_channels))
        image=tf.image.resize(image,[2*img_height,2*img_width])
        
        # std
        std=parsed_example['std']
        std=tf.image.decode_png(std,channels=nb_channels)
        std=(tf.cast(std,tf.float32)/127.5) -1
        std=tf.reshape(std,(img_height,img_width,nb_channels))
        std=tf.image.resize(std,[2*img_height,2*img_width])
        std = tf.where(std> threshold, 1.0, 0.0)
        
        # label
        label=parsed_example['label']
        label = tf.strings.to_number(tf.strings.split(label), out_type=tf.float32)
        label = tf.reshape(label,(pos_max,))
        
        # position
        pos=tf.range(0,pos_max)
        pos=tf.cast(pos,tf.int32) 
        
        return image,std,pos,label

    
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
for images,stds,poss,labels in train_ds.take(1):
    print("image")
    data=np.squeeze(images[0])
    plt.imshow(data)
    plt.show()    
    print("---------------------------------------------------------------")
    _label=labels[0].numpy()
    print(_label)
    text="".join([vocab[int(c)] for c in _label if vocab[int(c)] not in ["pad","sep"]])
    print("label :",text)
    print("---------------------------------------------------------------")
    print('Batch Shape:',images.shape)
    print("---------------------------------------------------------------")
    print("Positional encoding:",poss[0])
    print("std")
    data=np.squeeze(stds[0])
    plt.imshow(data)
    plt.show()
    print("---------------------------------------------------------------")

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

#----------------------------------------------
# discriminator downsample block
#----------------------------------------------
def downsample(filters, size, apply_norm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
  if apply_norm: result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result


#--------------------------------------------------------
# metrics and losses
#--------------------------------------------------------

def recognizer_accuracy(y_pred, y_true):
    accuracies = tf.equal(tf.cast(y_true,tf.int64), tf.argmax(y_pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(y_true,pad_value))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# loss
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
rec_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def recognizer_loss(pred, real):
    mask = tf.math.logical_not(tf.math.equal(real, pad_value))
    loss_ = rec_loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target,lambda_value=100):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (lambda_value * l1_loss)
    return total_gen_loss


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

def build_discriminator():
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004). """
    
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[64,512, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[64,512, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  
    down1 = downsample(64, 4,False)(x)  
    down2 = downsample(128, 4)(down1)  
    down3 = downsample(256, 4)(down2)  
    down4 = downsample(512, 4)(down3)  
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,use_bias=False)(zero_pad1)  
    norm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  
    last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2) 
    return tf.keras.Model(inputs=[inp, tar], outputs=last)



with strategy.scope():
    recognizer=build_model()
    recognizer.load_weights(RECOGNIZER_WEIGHT_PATH)
    generator=sm.Unet(GENERATOR_BACKBONE,input_shape=(2*img_height,2*img_width,3), classes=3,encoder_weights=None)
    generator.load_weights(GENERATOR_WEIGHT_PATH)
    discriminator=build_discriminator()
    discriminator.load_weights(DISCRIMINATOR_WEIGHT_PATH)


class ApsisNetv2(tf.keras.Model):
    def __init__(self,
                 generator,
                 recognizer,
                 discriminator,
                 loss_factor=10,
                 threshold=0.5):
        super(ApsisNetv2, self).__init__()
        self.generator     = generator
        self.recognizer    = recognizer
        self.discriminator = discriminator
        self.loss_factor   = loss_factor
        self.threshold     = threshold
        
    def compile(self,
                gen_optimizer,
                disc_optimizer,
                loss_recognizer,
                loss_generator,
                loss_discriminator,
                acc_recognizer):
        super(ApsisNetv2, self).compile()
        
        self.opt_gen  = gen_optimizer
        self.opt_disc = disc_optimizer
        
        self.loss_rec   = loss_recognizer
        self.loss_gen   = loss_generator
        self.loss_disc  = loss_discriminator
        
        self.acc_rec    = acc_recognizer
        

    def train_step(self, batch_data):
        image,std,pos,gt= batch_data
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generated
            generated  = self.generator(image)
            
            # real output patch disc
            disc_real_output = self.discriminator([image, std], training=True)
            # generated output patch disc
            disc_gen_output = self.discriminator( [image, generated], training=True)
            # reconizer
            gen_resized  = tf.image.resize(generated,[img_height,img_width])
            gen_resized  = tf.where(gen_resized> self.threshold, 1.0, 0.0)
            pred         = self.recognizer({"image":gen_resized,"pos":pos},training=False)
            
            # loss
            loss_gen = self.loss_gen(disc_gen_output, generated, std)
            loss_disc = self.loss_disc(disc_real_output, disc_gen_output)
            loss_rec = self.loss_rec(pred,gt)
            loss=loss_gen+self.loss_factor*loss_rec
            # acc
            acc_rec=self.acc_rec(pred,gt)

            
        # calc gradients    
        gen_grads     = gen_tape.gradient(loss,self.generator.trainable_variables)
        disc_grads    = disc_tape.gradient(loss_disc,self.discriminator.trainable_variables)
        
        # apply
        self.opt_gen.apply_gradients(zip(gen_grads,self.generator.trainable_variables))
        self.opt_disc.apply_gradients(zip(gen_grads,self.generator.trainable_variables))
        

        return {"loss_gen"    : loss_gen,
                "loss_disc"   : loss_disc,  
                "loss_rec"    : loss_rec,
                "loss"        : loss,
                "char_acc": acc_rec}

    def test_step(self, batch_data):
        image,std,pos,gt= batch_data
        
        # generated
        generated  = self.generator(image,training=False)
        
        # real output patch disc
        disc_real_output = self.discriminator([image, std], training=False)
        # generated output patch disc
        disc_gen_output = self.discriminator( [image, generated], training=False)
        # reconizer
        gen_resized  = tf.image.resize(generated,[img_height,img_width])
        gen_resized  = tf.where(gen_resized> self.threshold, 1.0, 0.0)
        pred         = self.recognizer({"image":gen_resized,"pos":pos},training=False)
        
        # loss
        loss_gen = self.loss_gen(disc_gen_output, generated, std)
        loss_disc = self.loss_disc(disc_real_output, disc_gen_output)
        loss_rec = self.loss_rec(pred,gt)
        loss=loss_gen+self.loss_factor*loss_rec
        # acc
        acc_rec=self.acc_rec(pred,gt)
        
        return {"loss_gen"    : loss_gen,
                "loss_disc"   : loss_disc,  
                "loss_rec"    : loss_rec,
                "loss"        : loss,
                "char_acc": acc_rec}


lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.0001,
                                                 decay_steps=DECAY_STEPS,
                                                 alpha= 0.01)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

with strategy.scope():
    model = ApsisNetv2(generator,
                       recognizer,
                       discriminator)


model.compile(generator_optimizer,
              discriminator_optimizer,
              recognizer_loss,
              generator_loss,
              discriminator_loss,
              recognizer_accuracy)




#------------------------------------------------------------------
# callbacks 
#------------------------------------------------------------------

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(patience=40, 
                                                  verbose=1, 
                                                  mode = 'auto')

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self,model_dir):
        self.best = 0
        self.output_dir = model_dir

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs['val_char_acc']
        if metric_value > self.best:
            print(f"Loss Improved epoch:{epoch} from {self.best} to {metric_value}",end="#")
            self.best = metric_value
            save_path = os.path.join(self.output_dir, "generator_best.h5")
            self.model.generator.save_weights(save_path)
            save_path = os.path.join(self.output_dir, "discriminator_best.h5")
            self.model.discriminator.save_weights(save_path)
            print("Saved Weights")
    def set_model(self, model):
        self.model = model

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
curves.to_csv(f"history_gan_gen_rec_64batch_10eps.csv",index=False)