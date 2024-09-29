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

```bash
Epoch 1/2
401920/401920 [==============================] - ETA: 0s - loss: 0.0173 - C_acc: 0.9963Loss Improved epoch:0 from inf to 0.0036478808615356684#Saved Weights
401920/401920 [==============================] - 70076s 174ms/step - loss: 0.0173 - C_acc: 0.9963 - val_loss: 0.0036 - val_C_acc: 0.9992
Epoch 2/2
401920/401920 [==============================] - ETA: 0s - loss: 0.0022 - C_acc: 0.9997Loss Improved epoch:1 from 0.0036478808615356684 to 0.0015497811837121844#Saved Weights
401920/401920 [==============================] - 69944s 174ms/step - loss: 0.0022 - C_acc: 0.9997 - val_loss: 0.0015 - val_C_acc: 0.9996
```
* Original Training steps and training for 5 epochs from the previous step 
