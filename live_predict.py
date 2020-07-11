import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image as keras_img
import requests
import numpy as np

url = 'http://192.168.31.121:8080/shot.jpg'

IMG_SIZE = 25
CATEGORIES = ['with_mask','without_mask']

model = tf.keras.models.load_model('1594460762')


font = cv2.FONT_HERSHEY_SIMPLEX 
 # org 
org = (50, 50) 
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2


while True:
    img_resp_pic= requests.get(url)
    img_arr_pic= np.array(bytearray(img_resp_pic.content),dtype=np.uint8)
    img = cv2.imdecode(img_arr_pic, -1)[0:1000,0:1000]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    disp_img = img
    img = cv2.resize(img, (100,100))

    x = keras_img.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    prediction = model.predict([x])
   
    image = cv2.putText(disp_img, CATEGORIES[int(prediction)], org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 

    cv2.imshow('frame',image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


