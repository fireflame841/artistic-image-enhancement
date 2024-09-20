import sys
if len(sys.argv) != 2:
    print("Usage: python part2.py input_img.jpg")
    sys.exit(1)
input_image_path = sys.argv[1]

import numpy as np
import cv2
    
def median_cut(img, temp, depth):
    
    if len(temp) == 0:
        return 
        
    if depth == 0:
        # when optimal buckets have been created, we quantize the image
        r = temp[:,0].mean()
        g = temp[:,1].mean()
        b = temp[:,2].mean()
    
        for i in temp:
            image[i[3]][i[4]] = [r, g, b]
        return
    
    # finding the color space with highest range
    l = []
    l.append(temp[:,0].max() - temp[:,0].min())
    l.append(temp[:,1].max() - temp[:,1].min())
    l.append(temp[:,2].max() - temp[:,2].min())
    
    maximum = l.index(max(l))

    # sorting the pixels by color space with highest range
    temp = temp[temp[:,maximum].argsort()]
    
    # splitting the array along the median and calling the function recursively till depth is 0
    median = (len(temp)+1)//2
    
    median_cut(img, temp[0:median], depth-1)
    median_cut(img, temp[median:], depth-1)
    



#image = cv2.imread("C:\\Users\\ADITYA\\OneDrive\\Desktop\\IITD\\Acads\\sem5\\col783\\783assignments\\messi_final1.png")
image = cv2.imread(input_image_path)
depth = 8

#creating a 2d array which stores the values of coordinates of a pixel along with the rgb values
two_d = []
 
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        color = image[i][j]
        two_d.append([color[0],color[1],color[2],i, j]) 
        
two_d = np.array(two_d)

#applying median cut algorithm
median_cut(image, two_d,depth)
      

# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
median_image_filename = f"median_cut_image"
cv2.imwrite(median_image_filename, image)

# image = cv2.imread("C:\\Users\\ADITYA\\OneDrive\\Desktop\\IITD\\Acads\\sem5\\col783\\783assignments\\messi_final1.png")
image = cv2.imread(input_image_path)
def dithering(image):
    width, height = image.shape[1],image.shape[0]
    pixels = image
    
    
    # scanning from left to right, top to bottom, quantizing pixel values one by one
    for y in range(1,height - 1):
        for x in range(1, width - 1):
            old_pixel = pixels[y, x]
            # rounding the values to 0 or 255
            new_pixel = (old_pixel>127.5)*255 
            
            # calculating error
            error = old_pixel - new_pixel
            
            pixels[y, x] = new_pixel
            
            #the pixel to the right gets 7/16 of the error, the pixel below gets 5/16, and diagonally adjacent pixels get 3/16 and 1/16
            
            pixels[y, x + 1] += error * 7 / 16
            pixels[y + 1, x - 1] += error * 3 / 16
            pixels[y + 1, x] += error * 5 / 16
            pixels[y + 1, x + 1] += error * 1 / 16
    
    return pixels

# image = cv2.imread("C:\\Users\\ADITYA\\OneDrive\\Desktop\\IITD\\Acads\\sem5\\col783\\783assignments\\messi_final1.png")
image = cv2.imread(input_image_path)
depth = 5

#creating a 2d array which stores the values of coordinates of a pixel along with the rgb values
two_d = []
 
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        color = image[i][j]
        two_d.append([color[0],color[1],color[2],i, j]) 
        
two_d = np.array(two_d)

# applying median cut algorithm
median_cut(image, two_d,depth)
      
# applying floyd steinberg dithering
image = image/1.0
image = dithering(image)   

floyd_image_filename = f"floyd_steinberg_image"
cv2.imwrite(floyd_image_filename, image)
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()