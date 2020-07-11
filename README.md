# Face Mask Detection using TensorFlow and Keras API
![alt text](https://raw.githubusercontent.com/vvijayk1999/face-mask-detection/master/acc-loss.jpg)

## Usage
Dependancies : TensorFlow 

Unzip the dataset and execute 'train.py' script.
## Testing
This model correctly classified the below two out of sample images.<br><br>
![alt text](https://raw.githubusercontent.com/vvijayk1999/face-mask-detection/master/1861.jpg)<br>
<b>without_mask</b><br><br>
![alt text](https://raw.githubusercontent.com/vvijayk1999/face-mask-detection/master/2083.jpg)<br>
<b>with_mask</b>

## Model Summary
```console
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 98, 98, 16)        448
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 49, 16)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 47, 128)       18560
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 23, 23, 128)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 21, 128)       147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 10, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 128)         147584
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 1024)              2098176
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 1025
=================================================================
Total params: 2,413,377
Trainable params: 2,413,377
Non-trainable params: 0
_________________________________________________________________
```
## Model accuracy and loss per Epoch
```console
Epoch 1/50
29/29 - 8s - loss: 5.6702 - acc: 0.6042 - val_loss: 1.5418 - val_acc: 0.7556
Epoch 2/50
29/29 - 6s - loss: 1.0009 - acc: 0.7535 - val_loss: 0.5951 - val_acc: 0.8870
Epoch 3/50
29/29 - 6s - loss: 0.5571 - acc: 0.8488 - val_loss: 0.4532 - val_acc: 0.8882
Epoch 4/50
29/29 - 6s - loss: 0.5128 - acc: 0.8536 - val_loss: 0.4051 - val_acc: 0.8900
Epoch 5/50
29/29 - 6s - loss: 0.4671 - acc: 0.8609 - val_loss: 0.3874 - val_acc: 0.8894
Epoch 6/50
29/29 - 6s - loss: 0.4565 - acc: 0.8639 - val_loss: 0.3960 - val_acc: 0.8977
Epoch 7/50
29/29 - 6s - loss: 0.4212 - acc: 0.8753 - val_loss: 0.4509 - val_acc: 0.8436
Epoch 8/50
29/29 - 6s - loss: 0.4005 - acc: 0.8791 - val_loss: 0.3483 - val_acc: 0.8983
Epoch 9/50
29/29 - 6s - loss: 0.4054 - acc: 0.8766 - val_loss: 0.3447 - val_acc: 0.8960
Epoch 10/50
29/29 - 6s - loss: 0.4145 - acc: 0.8677 - val_loss: 0.3666 - val_acc: 0.8966
Epoch 11/50
29/29 - 6s - loss: 0.3616 - acc: 0.8883 - val_loss: 0.3432 - val_acc: 0.9025
Epoch 12/50
29/29 - 6s - loss: 0.3632 - acc: 0.8872 - val_loss: 0.3593 - val_acc: 0.8870
Epoch 13/50
29/29 - 6s - loss: 0.3606 - acc: 0.8888 - val_loss: 0.3948 - val_acc: 0.8532
Epoch 14/50
29/29 - 6s - loss: 0.3506 - acc: 0.8915 - val_loss: 0.3954 - val_acc: 0.8520
Epoch 15/50
29/29 - 6s - loss: 0.3412 - acc: 0.8929 - val_loss: 0.3161 - val_acc: 0.9055
Epoch 16/50
29/29 - 6s - loss: 0.3403 - acc: 0.8912 - val_loss: 0.3257 - val_acc: 0.8966
Epoch 17/50
29/29 - 6s - loss: 0.3432 - acc: 0.8950 - val_loss: 0.3371 - val_acc: 0.8936
Epoch 18/50
29/29 - 6s - loss: 0.3569 - acc: 0.8856 - val_loss: 0.3182 - val_acc: 0.9025
Epoch 19/50
29/29 - 6s - loss: 0.3394 - acc: 0.8964 - val_loss: 0.3160 - val_acc: 0.8971
Epoch 20/50
29/29 - 6s - loss: 0.3401 - acc: 0.8923 - val_loss: 0.3266 - val_acc: 0.8966
Epoch 21/50
29/29 - 6s - loss: 0.3386 - acc: 0.8977 - val_loss: 0.5394 - val_acc: 0.7961
Epoch 22/50
29/29 - 6s - loss: 0.3462 - acc: 0.8888 - val_loss: 0.3301 - val_acc: 0.8864
Epoch 23/50
29/29 - 6s - loss: 0.3270 - acc: 0.9029 - val_loss: 0.3920 - val_acc: 0.8466
Epoch 24/50
29/29 - 6s - loss: 0.3240 - acc: 0.8983 - val_loss: 0.3084 - val_acc: 0.9013
Epoch 25/50
29/29 - 6s - loss: 0.3209 - acc: 0.8983 - val_loss: 0.3060 - val_acc: 0.9061
Epoch 26/50
29/29 - 6s - loss: 0.3244 - acc: 0.9034 - val_loss: 0.2949 - val_acc: 0.9055
Epoch 27/50
29/29 - 6s - loss: 0.3100 - acc: 0.9034 - val_loss: 0.3023 - val_acc: 0.9073
Epoch 28/50
29/29 - 6s - loss: 0.3081 - acc: 0.9040 - val_loss: 0.3321 - val_acc: 0.8906
Epoch 29/50
29/29 - 6s - loss: 0.3259 - acc: 0.8994 - val_loss: 0.5011 - val_acc: 0.7919
Epoch 30/50
29/29 - 6s - loss: 0.3075 - acc: 0.9064 - val_loss: 0.2979 - val_acc: 0.9090
Epoch 31/50
29/29 - 6s - loss: 0.3275 - acc: 0.8929 - val_loss: 0.4149 - val_acc: 0.8597
Epoch 32/50
29/29 - 6s - loss: 0.3009 - acc: 0.9110 - val_loss: 0.2923 - val_acc: 0.9049
Epoch 33/50
29/29 - 6s - loss: 0.3096 - acc: 0.9034 - val_loss: 0.3224 - val_acc: 0.8924
Epoch 34/50
29/29 - 6s - loss: 0.3029 - acc: 0.9096 - val_loss: 0.3370 - val_acc: 0.8912
Epoch 35/50
29/29 - 6s - loss: 0.2973 - acc: 0.9099 - val_loss: 0.2846 - val_acc: 0.9126
Epoch 36/50
29/29 - 6s - loss: 0.3049 - acc: 0.9069 - val_loss: 0.2821 - val_acc: 0.9120
Epoch 37/50
29/29 - 6s - loss: 0.3158 - acc: 0.8991 - val_loss: 0.3055 - val_acc: 0.8995
Epoch 38/50
29/29 - 6s - loss: 0.3005 - acc: 0.9072 - val_loss: 0.2833 - val_acc: 0.9084
Epoch 39/50
29/29 - 6s - loss: 0.2921 - acc: 0.9115 - val_loss: 0.3299 - val_acc: 0.8936
Epoch 40/50
29/29 - 6s - loss: 0.3097 - acc: 0.9067 - val_loss: 0.3270 - val_acc: 0.8960
Epoch 41/50
29/29 - 6s - loss: 0.2884 - acc: 0.9123 - val_loss: 0.3450 - val_acc: 0.8805
Epoch 42/50
29/29 - 6s - loss: 0.3025 - acc: 0.9053 - val_loss: 0.2888 - val_acc: 0.9126
Epoch 43/50
29/29 - 6s - loss: 0.3105 - acc: 0.9050 - val_loss: 0.3195 - val_acc: 0.8960
Epoch 44/50
29/29 - 6s - loss: 0.2866 - acc: 0.9142 - val_loss: 0.2764 - val_acc: 0.9126
Epoch 45/50
29/29 - 6s - loss: 0.2938 - acc: 0.9123 - val_loss: 0.2785 - val_acc: 0.9126
Epoch 46/50
29/29 - 6s - loss: 0.2859 - acc: 0.9110 - val_loss: 0.3170 - val_acc: 0.8960
Epoch 47/50
29/29 - 6s - loss: 0.2947 - acc: 0.9023 - val_loss: 0.2820 - val_acc: 0.9090
Epoch 48/50
29/29 - 6s - loss: 0.2910 - acc: 0.9058 - val_loss: 0.3190 - val_acc: 0.8930
Epoch 49/50
29/29 - 6s - loss: 0.2898 - acc: 0.9104 - val_loss: 0.2786 - val_acc: 0.9144
Epoch 50/50
29/29 - 6s - loss: 0.2826 - acc: 0.9156 - val_loss: 0.4921 - val_acc: 0.8460
```
