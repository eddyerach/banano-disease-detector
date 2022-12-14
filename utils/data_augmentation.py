from itertools import combinations, permutations
from itertools import permutations
import cv2
import numpy as np
import argparse
import os
import glob 
# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness for a dataset.')
parser.add_argument('--input', help='Path to images folder.', default='.')


def aplica_brillo(input, brightness):
    print(f'In brillo')
    #carpetas = ['test', 'train']
    #for carpeta in carpetas:
    path_1 = input + '/'
    print(f'path_1: {path_1}')
    files = list(next(os.walk(path_1)))
    #print(f'archivos: {files}')
    for file in files[2]:
        #print(f'file: {file}')
        if file:
            if file.find('jpg') != -1:
                img_path = path_1 + file
                print(f'aplicando ajuste brillo a: {img_path}')
                input_img = cv2.imread(img_path, 1)
                #brightness = brightness - 100
                #contrast = (contrast) / 100
                new_image = np.zeros(input_img.shape, input_img.dtype)
                new_image = cv2.convertScaleAbs(input_img, alpha=1, beta=brightness)
                print(f'**************escribiendo imagen: {img_path}')
                cv2.imwrite(img_path, new_image)

def aplica_contraste(input, contraste):
    print(f'En contraste')
    carpetas = ['test', 'train']
    for carpeta in carpetas:
        path_1 = input + '/' + carpeta + '/'
        files = list(next(os.walk(path_1)))
        for file in files[2]:
            if file:
                if file.find('.jpg') != -1:
                    img_path = path_1 + file
                    input_img = cv2.imread(img_path, 1)
                    print(f'aplicando ajuste contraste a: {img_path}')
                    #brightness = brightness - 100
                    #contrast = (contrast) / 100
                    new_image = np.zeros(input_img.shape, input_img.dtype)
                    new_image = cv2.convertScaleAbs(input_img, alpha=contraste, beta=0)
                    cv2.imwrite(img_path, new_image)

def aplica_saturacion(input, saturation):
    print(f'En saturacion')
    carpetas = ['test', 'train']
    for carpeta in carpetas:
        path_1 = input + '/' + carpeta + '/'
        files = list(next(os.walk(path_1)))
        for file in files[2]:
            if file:
                if file.find('.jpg') != -1:
                    img_path = path_1 + file
                    input_img = cv2.imread(img_path, 1)
                    new_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV).astype("float32")
                    #saturation = saturation / 10
                    (h, s, v) = cv2.split(new_image)
                    s = s*saturation
                    s = np.clip(s,0,255)
                    new_image = cv2.merge([h,s,v])   
                    new_image = cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_HSV2BGR)
                    cv2.imwrite(img_path,new_image)

def aplica_gamma(input, gamma):
    print(f'En gamma')
    carpetas = ['test', 'train']
    for carpeta in carpetas:
        path_1 = input + '/' + carpeta + '/'
        print(f'path_1: {path_1}')
        files = list(next(os.walk(path_1)))
        for file in files[2]:
            if file:
                if file.find('.jpg') != -1:
                    img_path = path_1 + file
                    input_img = cv2.imread(img_path, 1)
                    #gamma = gamma/10
                    lookUpTable = np.empty((1,256), np.uint8)
                    for i in range(256):
                        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            
                    new_image = cv2.LUT(input_img, lookUpTable)
                    cv2.imwrite(img_path,new_image)


    
def apply_image_adjust(input):
    brillo = [40,-40]
    contraste = [1.4,0.6]
    saturacion = [2.4,0.7]
    gamma = [2.0,0.6]

    #Obtener los nombres de las carpetas
    path = input+'/'
    #print(f'path: {input}')
    #carpetas = list(next(os.walk(path))[1])
    carpetas = ['+brillo','-brillo']
    print(f'carpetas: {carpetas}')
    for carpeta in carpetas:
        if carpeta[1:] == 'brillo':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_brillo(path, brillo[0])
            else:
                aplica_brillo(path, brillo[1])   
        if carpeta[1:] == 'contraste':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_contraste(path, contraste[0])
            else:
                aplica_contraste(path, contraste[1])
        if carpeta[1:] == 'saturacion':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_saturacion(path, saturacion[0])
            else:
                aplica_saturacion(path, saturacion[1])   
        if carpeta[1:] == 'gamma':
            path = os.path.join(input, carpeta)
            if carpeta[0] == '+':
                aplica_gamma(path, gamma[0])
            else:
                aplica_gamma(path, gamma[1])
           
        


    

def funcBrightContrast(bright=0):
    bright = cv2.getTrackbarPos('bright', 'Test')
    contrast = cv2.getTrackbarPos('contrast', 'Test')
    saturation = cv2.getTrackbarPos('saturation', 'Test')
    gamma = cv2.getTrackbarPos('gamma', 'Test')
    effect = apply_brightness_contrast(img,bright,contrast, saturation, gamma)
    cv2.imshow('Effect', effect)


def apply_brightness_contrast(input_img, brightness, contrast, saturation, gamma):
    """
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    """
    #brillo y contraste
    brightness = brightness - 100
    contrast = (contrast + 100) / 100
    new_image = np.zeros(input_img.shape, input_img.dtype)
    new_image = cv2.convertScaleAbs(input_img, alpha=contrast, beta=brightness)
    #saturacion
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV).astype("float32")
    saturation = saturation / 10
    (h, s, v) = cv2.split(new_image)
    s = s*saturation
    s = np.clip(s,0,255)
    new_image = cv2.merge([h,s,v])   
    new_image = cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    #correccion gamma
    gamma = gamma/10
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    
    new_image = cv2.LUT(new_image, lookUpTable)
    cv2.putText(new_image,'B:{},C:{},S:{},G:{}'.format(brightness,contrast,saturation, gamma),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return new_image

"""
def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)
"""

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.input
    print(f'input: {path}')

    '''
    brillo = [-1,0,1]
    contraste = [-1, 0, -1]
    saturacion = [-1, 0, -1]
    gamma = [-1, 0, -1]
    '''
    apply_image_adjust(path)


    '''
    original = cv2.imread("test.png", 1)
    img = original.copy()
    cv2.namedWindow('Test',1)
    bright = 100
    contrast = 0
    saturation = 20
    gamma = 10
    #alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    #Brightness value range -255 to 255
    #Contrast value range -127 to 127
    cv2.createTrackbar('bright', 'Test', bright, 200, funcBrightContrast)
    cv2.createTrackbar('contrast', 'Test', contrast, 200, funcBrightContrast)
    cv2.createTrackbar('saturation', 'Test', saturation, 70, funcBrightContrast)
    cv2.createTrackbar('gamma', 'Test', gamma, 250, funcBrightContrast)
    funcBrightContrast(0)
    cv2.imshow('Test', original)
    '''
cv2.waitKey(0)