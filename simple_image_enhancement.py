import cv2
import numpy as np
import math
import sys
if len(sys.argv) != 2:
    print("Usage: python part1.py input_img.jpg")
    sys.exit(1)
input_image_path = sys.argv[1]
img = cv2.imread(input_image_path)
# img1 = cv2.imread('C:/Users/Atharv/OneDrive/Desktop/ref.jfif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img)
img = img/255.0
rows = img.shape[0]
cols = img.shape[1]
shp=(rows, cols, 3)
iv=np.zeros(shp)

#converting to hsi
def gethsi(im):
    hsi=np.zeros(shp)
    a = np.array([[1/3,1/3,1/3],[-1*math.sqrt(6)/6,-1*math.sqrt(6)/6,math.sqrt(6)/3],[1/math.sqrt(6),-1*2/math.sqrt(6),0]])
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            iv[i][j] = np.matmul(a,im[i][j].reshape((3,1))).reshape(3)
            
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            v1 = iv[i][j][1]
            v2 = iv[i][j][2]
            if v1 == 0:
                h = 0
            else:
                h = math.atan(v2/v1)
            s = math.sqrt((v1*v1) + (v2*v2))
            hsi[i][j][0],hsi[i][j][1],hsi[i][j][2] = h,s,iv[i][j][0]
    return hsi
        
hsi = gethsi(img)/[math.pi/2,1.0,1.0]

#r-map generation
r1=np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        r1[i][j]=(1+hsi[i][j][0])/(1+hsi[i][j][2])

mr1=np.min(r1)
mxr1=np.max(r1)
#scaling the r-map
for i in range(rows):
    for j in range(cols):
        r1[i][j]=(r1[i][j]-mr1)*(255/mxr1)

def compute_otsu_criteria(im, th):
    """Otsu's method to compute criteria."""
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1

im = r1
# testing all thresholds from 0 to the maximum of the image
threshold_range = range(int(np.max(im))+1)
criterias = [compute_otsu_criteria(im, th) for th in threshold_range]

# best threshold is the one minimizing the Otsu criteria
best_threshold = threshold_range[np.argmin(criterias)]
best_threshold

# shadow map generation
s=np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        s[i][j]=1
        if(r1[i][j]>best_threshold):
            s[i][j]=0

# shadow image generation
shadow_img=cv2.imread(input_image_path)
for i in range (rows):
    for j in range(cols):
        if(s[i][j]==0):
            shadow_img[i][j][0]*=0.6
            shadow_img[i][j][1]*=0.6
            shadow_img[i][j][2]*=0.6

gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian(x, sigma):
    '''This function defines a Gaussian function, which is used later in the bilateral filter. 
    It takes two arguments: x, which represents the input value, and sigma, which controls the
    spread or width of the Gaussian distribution.The function calculates the Gaussian value for
    a given x and sigma using the formula for a Gaussian distribution.'''
    return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))

def bilateral_filter(image, diameter, sigma_color, sigma_space):
    output = np.zeros_like(image, dtype=np.float64)

    pad_size = diameter // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
    # Here, we calculate the padding size required for the image based on the filter diameter.
    # We pad the input image using cv2.copyMakeBorder to ensure that we can apply the filter to pixels near the image borders. 
    # The cv2.BORDER_CONSTANT flag specifies that we want to pad with a constant value (typically zero).

    color_coeff = -0.5 / (sigma_color ** 2)
    space_coeff = -0.5 / (sigma_space ** 2)
    # These lines compute coefficients used in the bilateral filter. 
    # color_coeff and space_coeff are precomputed constants based on the provided sigma_color and sigma_space values. 
    # They will be used to calculate color and spatial similarity weights later in the filter.

    for y in range(pad_size, padded_image.shape[0] - pad_size):
        for x in range(pad_size, padded_image.shape[1] - pad_size):
            center_pixel = padded_image[y, x]
            filtered_value = 0.0
            normalization = 0.0
            for j in range(-pad_size, pad_size + 1):
                for i in range(-pad_size, pad_size + 1):
                    current_pixel = padded_image[y + j, x + i]
                    color_diff = center_pixel - current_pixel
                    color_similarity = np.exp(color_coeff * color_diff ** 2)
                    # Within these loops, we calculate the color difference between the current pixel and the center pixel. 
                    # We then compute the color similarity based on the color difference using the Gaussian function defined earlier.

                    spatial_diff = np.sqrt(i ** 2 + j ** 2)
                    spatial_similarity = np.exp(space_coeff * spatial_diff ** 2)
                    # We also calculate the spatial difference between the current pixel and 
                    # the center pixel's position and compute the spatial similarity based on 
                    # this difference using the Gaussian function.

                    weight = color_similarity * spatial_similarity
                    filtered_value += current_pixel * weight
                    normalization += weight
                    # We calculate the weight for the current pixel by multiplying the color and spatial similarities. 
                    # We then update the filtered value by multiplying the current pixel's value with this weight 
                    # and update the normalization factor by adding the weight.
                    
            output[y - pad_size, x - pad_size] = filtered_value / normalization

    return output

diameter = 7
sigma_color = 20
sigma_space = 20
filtered_image = bilateral_filter(gray_image, diameter, sigma_color, sigma_space) #reduces noise

'''This function takes three arguments:

bi_im: The input image on which edge detection is to be performed.
vertical_filter: The vertical edge detection filter/kernel.
horizontal_filter: The horizontal edge detection filter/kernel.'''
def detect_edges(bi_im, vertical_filter, horizontal_filter):
        kernel_width = vertical_filter.shape[0]//2
        grad_ = np.zeros(bi_im.shape)
        # Here, kernel_width is calculated as half the height of the vertical_filter, assuming that the filter is square. 
        # grad_ is initialized as an array of zeros with the same shape as the input image bi_im. 
        # This array will store the gradient magnitude at each pixel location.

        bi_im = np.pad(bi_im, pad_width= ([kernel_width, ], [kernel_width, ]), 
        # This line pads the input image bi_im with zeros on all sides. 
        # The amount of padding is determined by kernel_width, ensuring that the convolution operation can be applied to 
        # all pixels without going out of bounds.

        mode= 'constant', constant_values= (0, 0))

        for i in range(kernel_width, bi_im.shape[0] - kernel_width):
            for j in range(kernel_width, bi_im.shape[1] - kernel_width):
                x = bi_im[i - kernel_width: i + kernel_width + 1, j - kernel_width: j + kernel_width + 1]
                # Here, x is a local region (window) from the padded image, which corresponds to the same size as the vertical_filter.
                x = x.flatten() * vertical_filter.flatten()
                sum_x = x.sum()
                # This code flattens both x and the vertical_filter, element-wise multiplies them, and then calculates the sum.
                # This is effectively performing convolution between the x region and the vertical_filter.
                y = bi_im[i - kernel_width: i + kernel_width + 1, j - kernel_width: j + kernel_width + 1]
                y = y.flatten() * horizontal_filter.flatten()
                sum_y = y.sum()
                # This code does the same as above but for y and the horizontal_filter.
        
                grad_[i - kernel_width][j - kernel_width] = np.sqrt(sum_x**2 + sum_y**2)
        bi_im = grad_
        return bi_im

vertical_filter = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]) # SobelY operator
horizontal_filter = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]) # SobelX operator
edge_map=detect_edges(filtered_image, vertical_filter, horizontal_filter) # we pass the bilateral filtered image in the function

# line draft generation
ld=np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        ld[i][j]=1
        if(edge_map[i][j]>0.3): # Threshold selected as 0.3
            ld[i][j]=0

imglab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) #conversion to LAB space
L, A, B = cv2.split(imglab)
neutral_lightness = np.full_like(L, np.median(L))  #neutral lightness channel generation
chromatic_map = cv2.merge((neutral_lightness, A, B)) #chromatic_map generation
c_m = cv2.cvtColor(chromatic_map, cv2.COLOR_LAB2BGR) #reverting to BGR space

rho = 0.2
enhanced_si = np.zeros_like(shadow_img, dtype=np.uint8)

for x in range(shadow_img.shape[0]):
    for y in range(shadow_img.shape[1]):
        cm_color = c_m[x, y]

        enhancement_factor = 1 + np.tanh(rho * (np.mean(cm_color) - 128)) / 2

        enhanced_channel = shadow_img[x, y] * enhancement_factor
        enhanced_channel = np.clip(enhanced_channel, 0, 255).astype(np.uint8)

        enhanced_si[x, y] = enhanced_channel

# alpha=2.5
# beta=5
# adjusted = cv2.convertScaleAbs(enhanced_si, alpha=alpha, beta=beta)
def saturation_correction(im, s_scale):
    h_s_i=cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    # h_s_i=gethsi(im)
    # for i in range(0, h_s_i.shape[0]):
    #     for j in range(0, h_s_i.shape[1]):
    #         h_s_i[i][j][1]*=s_scale
    #         h_s_i[i][j][0]*=s_scale*math.pi/2
    # h_s_i=(h_s_i).round().astype(np.uint8)
    h, s, i = cv2.split(h_s_i)
    s = np.clip(s * s_scale, 0, 255).astype(np.uint8)
    adjusted_hsi_image = cv2.merge((h, s, i))
            
    return cv2.cvtColor(adjusted_hsi_image, cv2.COLOR_HSV2BGR_FULL)

adjusted=saturation_correction(enhanced_si, 2.5)
adjusted = np.clip(adjusted * 2.5, 0, 255).astype(np.uint8)

final=adjusted
for i in range(0, rows):
    for j in range(0, cols):
        if(ld[i][j]==0):
            final[i][j][0]*=0.2
            final[i][j][1]*=0.2
            final[i][j][2]*=0.2

output_image_filename = f"Artistic_rendered_image"
cv2.imwrite(output_image_filename, final)

