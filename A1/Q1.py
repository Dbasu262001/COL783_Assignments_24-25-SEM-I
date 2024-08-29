# %%
import cv2
import pandas as pd
import numpy as np
import os
# function to read image
def read_image(filename : str)-> np.array:
    return cv2.imread(filename)

# converting image to grayscale image
def get_graycale(image:np.array):
    weights= [0.3, 0.58, 0.12]
    gray = np.dot(img[..., :3], weights)
    return np.round(gray).astype(np.int16)

img = read_image("data/cse-logo.png")
img.shape

# %%
my_image = cv2.imread("data/eagle.jpeg", cv2.IMREAD_COLOR)
my_image = np.dot(my_image[..., :3], [0.3, 0.58, 0.12])
my_image = np.round(my_image).astype(np.uint8)
M = max(my_image.shape[0],my_image.shape[1])
Scaling_factor = int(.25*M)
Scaling_factor

# %%

gray_image = get_graycale(img)


# %%
def calculate_histogram(image:np.array):
    array = image.flatten()
    histogram = np.array([0]*256)
    for i in range(array.shape[0]):
        histogram[array[i]]+=1
    return histogram
histogram =calculate_histogram(gray_image)
# print(histogram)
r_star =np.argmax(histogram)
print("r* = ",r_star)

# %%
t=10
image_a = np.where(np.abs(gray_image - r_star) <= 10, 1, 0).astype(np.uint8)
cv2.imwrite("output_images/Q1/Q1_image_a.png",image_a*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
print("t = ",t)


# %%
# cv2.imshow('Q1(a) sep background & foreground', image_a*255)
# cv2.waitKey(1500)
# cv2.destroyAllWindows()

# %%
#************** Q1 B************
############################################
#Nearest neighbour interpolation
def get_rounded(scale_factor:float,val:float)->int:
    if(scale_factor < 1):
        return int(val/scale_factor)
    else:
        return int(val*scale_factor)

def Nearest_neighbour_interpolation(image:np.array,size:int):
    image_shape = image.shape
    N = int(.25*size)
    row_scale = image_shape[0]/N
    col_scale = image_shape[1]/N
    resized_image = np.zeros((N,N),dtype=np.int16)
    for i in range(N):
        for j in range(N):
            i_old = int(i*row_scale)
            j_old = int(j*col_scale)
            resized_image[i,j] = image[i_old,j_old]

    return resized_image



# %%
nni_image =Nearest_neighbour_interpolation(image_a,M)
cv2.imwrite("output_images/Q1/Q1_nearest_neigh_image_a.png",nni_image*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imshow('nearest_neigh_image_a', nni_image*255)  
# cv2.waitKey(1500)
# cv2.destroyAllWindows()
# print(nni_image.shape)

# %%
nni_img = Nearest_neighbour_interpolation(gray_image,M)
cv2.imwrite("output_images/Q1/Q1_nearest_neigh_source.png",nni_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imshow('nearest_neigh_image_source', nni_img)  
# cv2.waitKey(1500)
# cv2.destroyAllWindows()

# %%
#******** bilinear interpolation**********

def bilinear_interpolation(image: np.array, size: tuple):
    old_height, old_width = image.shape
    N = int(0.25*size)
    row_scale = (old_height) / N
    col_scale = (old_width) / N
    
    resized_image = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            i_old = i * row_scale
            j_old = j * col_scale
            
            x1 = int(np.floor(i_old))
            y1 = int(np.floor(j_old))
            x2 = min(x1 + 1, old_height - 1)
            y2 = min(y1 + 1, old_width - 1)
            
            f11 = image[x1, y1]
            f12 = image[x1, y2]
            f21 = image[x2, y1]
            f22 = image[x2, y2]
            
            d_i = i_old - x1
            d_j = j_old - y1
            
            linear1 = (1 - d_j) * f11 + d_j * f12
            linear2 = (1 - d_j) * f21 + d_j * f22
            
            resized_image[i, j] = (1 - d_i) * linear1 + d_i * linear2
    
    return resized_image


    

# %%
bi_img_src = bilinear_interpolation(gray_image,M)
cv2.imwrite("output_images/Q1/Q1_bilinear_source.png",bi_img_src,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imshow('Bilinear on source', bi_img_src*255)  
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
np.max(bi_img_src)

# %%
bi_img_a = bilinear_interpolation(image_a,M)
cv2.imwrite("output_images/Q1/Q1_bilinear_image_a.png",bi_img_a*255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imshow('Bilinear on image a', bi_img_src*255)  
# cv2.waitKey(1000)
# cv2.destroyAllWindows()


# %%
#**************Q1 C *******************
############################################
logo = bi_img_a
M,N = my_image.shape[0],my_image.shape[1]
logo_size = bi_img_a.shape[0]
start_i,start_j = M-logo_size,0
for i in range(logo_size):
    for j in range(logo_size):
        x = start_i+i
        y = start_j +j
        p = (1-logo[i,j])*255
        my_image[x,y] = max(p,my_image[x,y])




# %%
cv2.imwrite("output_images/Q1/Q1_logo_paste_a.png",my_image,[cv2.IMWRITE_PNG_COMPRESSION, 0])



