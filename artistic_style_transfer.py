import cv2
import numpy as np
import math
import sys

if len(sys.argv) != 3:
    print("Usage: python part3.py ref_img.jpg grey_img.jpg")
    sys.exit(1)
ref_image_path = sys.argv[1]
grey_image_path = sys.argv[2]

import cv2
import numpy as np
import math

# img = cv2.imread('C:/Users/Atharv/OneDrive/Desktop/greyscale.jpg')
img = cv2.imread(grey_image_path)
# if image is 2 dimensional, we take the L in labspace as intensity and keep A and B as 0
if len(img.shape) == 2:    
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

# if image is 3 dimensional, we convert it to lab
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
rows = img.shape[0]
cols = img.shape[1]

ref = cv2.imread(ref_image_path)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
swatch_size = 5
ref_swatches = 2
# manually selecting swatches in reference image, storing mean of L values and coordinates of the swatch in l_ref
l_ref = np.zeros((ref_swatches,3))
l_ref[0] = [ref[500:505,800:805].mean(),500,800]
l_ref[1] = [ref[50:55,800:805].mean(),50,800]

# creating a 2D array to store the mean values of L for a swatch
l = np.zeros((rows//swatch_size+1,cols//swatch_size+1))



# filling l
for i in range(0,rows, swatch_size):
    for j in range(0,cols,  swatch_size):
        l[i//swatch_size][j//swatch_size] = img[i:i+swatch_size , j:j+swatch_size , 0].mean()
        

# substituting L values of each swatch to the closest swatch in the reference image 

for i in range(0,rows//swatch_size):
    for j in range(0,cols//swatch_size):
        minimum = l_ref[0][0]
        index = 0
        for k in range(0,ref_swatches):
            if abs(l[i][j] - l_ref[k][0]) < minimum:
                index = k
        x = int(l_ref[index,1])
        y = int(l_ref[index,2])
        img[i*swatch_size:i*swatch_size+swatch_size, j*swatch_size:j*swatch_size+swatch_size,1:] = ref[x:x+swatch_size,y:y+swatch_size,1:] 

# converting back to rgb
img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

output_image_filename = f"artistic_transfered_image"
cv2.imwrite(output_image_filename, img)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()