from PIL import Image
import numpy as np

def Predict_Image(img, model):
    IMAGE_WIDTH=128
    IMAGE_HEIGHT=128
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    
    img = img.resize(IMAGE_SIZE)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255
    result = model.predict(img)
    
    return result

