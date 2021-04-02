from PIL import Image
import numpy as np

def Predict_Woman_Man(img, model):
    IMAGE_WIDTH=96
    IMAGE_HEIGHT=96
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    
    img = img.resize(IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    
    return result

