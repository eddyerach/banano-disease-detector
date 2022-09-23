import cv2 
import glob

def zero_padding(img):
    dimensions = img.shape
    # height, width, number of channels in image
    height0 = img.shape[0]
    width0 = img.shape[1]
    right = crop_width - width0
    bottom = crop_height - height0
    print(f'bottom: {bottom} right: {right}')
    img_pad = cv2.copyMakeBorder(img, 0, 0, bottom, right, cv2.BORDER_CONSTANT, (0,0,0))
    return img_pad

path =      '/exterior/conteo_plantas/mosaicos/mosaicos/*jpg'
dest_path = '/exterior/conteo_plantas/mosaicos/crops'
images = glob.glob(path)
crop_width = 2000
crop_height = 2000
#print(f'images: {images}')
for im in images:
    img = cv2.imread(im)
    # height, width, number of channels in image
    height0 = img.shape[0]
    width0 = img.shape[1]
    rows = width0 // crop_width
    cols = height0 // crop_height
    y0 = 0
    x0 = 0
    acu = 0
    print(f'Image: {im}')
    for row in range(0,rows + 1):
      for col in range(0,cols + 1):
        x1 = x0 + crop_width
        y1 = y0 + crop_height
        if row == rows:
          y1 = y0 + (height0 - y0)
        if col == cols:
          x1 = x0 + (width0 - x0)

        crop_img = img[y0:y1, x0:x1]
        name_crop = im.split('.')[0] + '_' + str(row) + '-' + str(col) + '.jpg'
        name_dest = dest_path + '/' + name_crop.split('/')[-1]  
        if row == rows or col == cols:
          crop_img = zero_padding(crop_img)
        try:
          print(f'Saving: {name_dest}')
          cv2.imwrite(name_dest, crop_img)
        except:
          print(f'La imagen: ({row},{col}) no pudo ser generada')
        x0 = x1
        acu += 1
      x0 = 0
      y0 = y1

