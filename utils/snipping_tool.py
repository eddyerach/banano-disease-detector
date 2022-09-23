import cv2
from os import walk
import io 
import json
import os
import glob

img_list = glob.glob(('/exterior/conteo_plantas/mosaicos/mosaicos/*'))
path = '/exterior/conteo_plantas/mosaicos/mosaicos'
def zero_padding(img):
    dimensions = img.shape
    # height, width, number of channels in image
    height0 = img.shape[0]
    width0 = img.shape[1]
    right = crop_width - width0
    bottom = crop_height - height0
    img_pad = cv2.copyMakeBorder(img, 0, 0, bottom, right, cv2.BORDER_CONSTANT, (0,0,0))
    return img_pad

def get_image(path):
    img = cv2.imread(path)
    return img 

def cut_images(dims, padding):
    crop_width = dims[0]
    crop_height = dims[1]
    dest_path = os.path.join(os.path.split(path)[0] ,'crops','cut_' + name)
    try:
        os.mkdir(dest_path)
        print('Creating folder: ', dest_path)
    except: 
        print(f'The folder: {dest_path} already exists!')

    os.chdir(dest_path)
    os.system('rm *')
    if padding == 'zero':
        cut_zero_padding(dest_path)

def cut_zero_padding(dest_path):
    #global img_list
    images_list = img_list
    for image in images:
      #print(f"Loading image {path_img}")
      img = get_image(image)
      #cv2_imshow(img)
      dimensions = img.shape
      # height, width, number of channels in image
      height0 = img.shape[0]
      width0 = img.shape[1]
      #compute number of images that can be cropped from the image
      rows = width0 // crop_width
      cols = height0 // crop_height
      y0 = 0
      x0 = 0
      acu = 0
      print(f'Image: {image}')
      for row in range(0,rows + 1):
        for col in range(0,cols + 1):
          x1 = x0 + crop_width
          y1 = y0 + crop_height

          if row == rows:
            y1 = y0 + (height0 - y0)
          if col == cols:
            x1 = x0 + (width0 - x0)
          
          crop_img = img[y0:y1, x0:x1]
          name_crop = image.split('.')[0] + '_' + str(row) + '-' + str(col) + '.jpg'
          
          if row == rows or col == cols:
            crop_img = zero_padding(crop_img)
          try:
            cv2.imwrite(name_crop, crop_img)
          except:
            print(f'La imagen: ({row},{col}) no pudo ser generada')
          x0 = x1
          acu += 1
        x0 = 0
        y0 = y1

cut_images((2000,2000), 'zero')