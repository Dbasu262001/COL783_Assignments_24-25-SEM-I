# %%
import numpy as np
import pandas as pd
import cv2
import os

image3 = cv2.imread("data/memorial.hdr",cv2.IMREAD_ANYDEPTH)


# %%
weights= [0.3, 0.59, 0.11]

gray_image3 = np.dot(image3,weights) #converting to gray scale image
gray_image3_show = 255*(gray_image3 - np.min(gray_image3))/(np.max(gray_image3)-np.min(gray_image3))
cv2.imwrite('output_images/Q3/Q3_a_original.png', gray_image3_show.astype(np.uint8),[cv2.IMWRITE_PNG_COMPRESSION, 0])

# %%
######Filters 
def gaussian_kernel_1d(size:int,sigma:float)->np.array:
    distance = np.square(np.linspace(-(size//2),size//2,size))
    kernel = np.exp(-0.5 * distance / np.square(sigma) )
    return kernel/np.sum(kernel)
###Convolution 1D
def convolve_1D(A:np.array,kernel:np.array)->np.array:
    C=[]
    for row in A:
        c_rows = np.convolve(row,kernel,mode='same')
        C.append(c_rows)
    return np.array(C)
def full_convolution(A:np.array,kernel:np.array)->np.array:
    A1 = convolve_1D(A,kernel).T
    result = convolve_1D(A1,kernel).T
    return result

# %%
#Gaussian filter on image
def gaussian_filter(image:np.array,sigma:int)->np.array:
    size=2*sigma+1
    kernel = gaussian_kernel_1d(size=size,sigma=sigma)
    filtered_image = full_convolution(image,kernel=kernel)
    return filtered_image


# %%
#********** Q3 A *******************
#image with sigma 2
gaussian_sigma2 = gaussian_filter(gray_image3,2)
gaussian_sigma2 = (255*(gaussian_sigma2 - np.min(gaussian_sigma2))/(np.max(gaussian_sigma2)-np.min(gaussian_sigma2))).astype(np.uint8)
#####save image
cv2.imwrite('output_images/Q3/Q3_a_sigma_2.png', gaussian_sigma2,[cv2.IMWRITE_PNG_COMPRESSION, 0])


# %%
gaussian_sigma8 = gaussian_sigma2 = gaussian_filter(gray_image3,8)
gaussian_sigma8 = (255*(gaussian_sigma8 - np.min(gaussian_sigma8))/(np.max(gaussian_sigma8)-np.min(gaussian_sigma8))).astype(np.uint8)
#####save image
cv2.imwrite('output_images/Q3/Q3_a_sigma_8.png', gaussian_sigma8,[cv2.IMWRITE_PNG_COMPRESSION, 0])
#******************************************

# %%
#***********Q3 B****************************
f_bar = np.log(gray_image3) # Log image 
f_bar_show = (255*(f_bar - np.min(f_bar)) / (np.max(f_bar) - np.min(f_bar)) ).astype(np.uint8)
cv2.imwrite('output_images/Q3/Q3_b_f_bar_log.png', f_bar_show,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# ************

# %%
g_bar = gaussian_filter(f_bar,2) # applying low pass filter on log image
g_bar_show =  (255*(g_bar - np.min(g_bar) )/ (np.max(g_bar) - np.min(g_bar)) ).astype(np.uint8)
cv2.imwrite('output_images/Q3/Q3_b_g_bar_low.png', g_bar_show,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# ************

# %%
#contrast reduction  (lowest value to 1 scaling)
g_bar_reduced = g_bar/np.min(g_bar)
g_bar_reduced_show = (255*(g_bar_reduced - np.min(g_bar_reduced) )/ (np.max(g_bar_reduced) - np.min(g_bar_reduced)) ).astype(np.uint8)
cv2.imwrite('output_images/Q3/Q3_b_g_bar_low_reduced.png', g_bar_reduced_show,[cv2.IMWRITE_PNG_COMPRESSION, 0])



# %%
h_bar = f_bar - g_bar # applying high pass filter on log image
h_bar_show = (255* (h_bar - np.min(h_bar)) / (np.max(h_bar) - np.min(h_bar))  ).astype(np.uint8)
cv2.imwrite("output_images/Q3/Q3_b_h_bar_high.png",h_bar_show,[cv2.IMWRITE_PNG_COMPRESSION,0])
# ************

# %%
recomposed_image = np.exp(h_bar + g_bar_reduced) # Applying exponent 
recomposed_image = (255*(recomposed_image - np.min(recomposed_image) )/ (np.max(recomposed_image) - np.min(recomposed_image) )).astype(np.uint8)
cv2.imwrite("output_images/Q3/Q3_b_recomposed.png",recomposed_image,[cv2.IMWRITE_PNG_COMPRESSION,0])

# %%
#**************************************
#*********** Q3 C ************
def gaussian_kernel_2D(size:int,sigma)->np.array:
    if(size % 2 ==0):
        raise("Enter odd value")
    x, y = np.meshgrid(np.arange(-(size//2), (size//2) + 1), np.arange(-(size//2), (size//2) + 1))
    kernel = np.exp( -0.5* (np.square(x) + np.square(y))/(np.square(sigma)))
    gaussian_kernel = kernel/np.sum(kernel)
    return gaussian_kernel

    
def bilater_filter(image1:np.array,sigma_s,sigma_r):
    size = 2*sigma_s +1
    gauss_spatial = gaussian_kernel_2D(size=size,sigma=sigma_s)
    padding_size = size//2
    padded_image = np.pad(image1,padding_size)
    image = np.zeros(image1.shape)
    m,n = image.shape[0],image.shape[1]
    for i in range(m):
        for j in range(n):
            f_q = padded_image[i:i+size,j:j+size]
            fp_fq = padded_image[i:i+size,j:j+size] - image[i,j]
            gauss_range = np.exp(-0.5*np.square(fp_fq)/np.square(sigma_r))
            gaussian_range = gauss_range/np.sum(gauss_range)
            image[i,j] =np.sum(gauss_spatial*f_q*gaussian_range)
    return image 


    

# %%
sigma_pairs=[[2,2],[2,8],[2,8]]
for i in range(3):
    sig_s = sigma_pairs[i][0]
    sig_r = sigma_pairs[i][1]
    bilat_im = bilater_filter(gray_image3,sigma_s=sig_s,sigma_r=sig_r)
    bilat_img_scaled = (255*(bilat_im - np.min(bilat_im))/(np.max(bilat_im) - np.min(bilat_im)))
    name = str(sig_s) + "_"+str(sig_r) +".png"
    cv2.imwrite("output_images/Q3/Q3_c_bilateral_"+name,bilat_img_scaled,[cv2.IMWRITE_PNG_COMPRESSION,0])

# %%
#***********************************
#********** Q3  d********
f_bar = np.log(gray_image3)
f_bar_show = (255*(f_bar - np.min(f_bar)) / (np.max(f_bar) - np.min(f_bar)) ).astype(np.uint8)
cv2.imwrite('output_images/Q3/Q3_d_f_bar_log.png', f_bar_show,[cv2.IMWRITE_PNG_COMPRESSION, 0])

# %%
sigma_list = [[2,2],[2,3],[3,2]]

# %%
# Taking inverse of log i.e exponent
for x in sigma_list:
    sigma_s = x[0]
    sigma_r = x[1]
    low_pass = bilater_filter(f_bar,sigma_s=sigma_s,sigma_r=sigma_r)
    hig_pass = f_bar - low_pass
    low_pass_reduced = low_pass/np.min(low_pass)
    bilateral_recomp_img = np.exp( hig_pass + low_pass)
    display_image = (255*(bilateral_recomp_img - np.min(bilateral_recomp_img))/ (np.max(bilateral_recomp_img) - np.min(bilateral_recomp_img)) ).astype(np.uint8)
    lst = str(sigma_s) +'_'+str(sigma_r)+".png"
    cv2.imwrite('output_images/Q3/Q3_d_exp_recomp_img'+lst, display_image,[cv2.IMWRITE_PNG_COMPRESSION, 0])


# %%
##########images without taking log inverse
for x in sigma_list:
    sigma_s = x[0]
    sigma_r = x[1]
    low_pass = bilater_filter(f_bar,sigma_s=sigma_s,sigma_r=sigma_r)
    hig_pass = f_bar - low_pass
    low_pass_reduced = low_pass/np.min(low_pass)
    bilateral_recomp_img = hig_pass + low_pass
    display_image = (255*(bilateral_recomp_img - np.min(bilateral_recomp_img))/ (np.max(bilateral_recomp_img) - np.min(bilateral_recomp_img)) ).astype(np.uint8)
    lst = str(sigma_s) +'_'+str(sigma_r)+".png"
    cv2.imwrite('output_images/Q3/Q3_d_log_recomp_img'+lst, display_image,[cv2.IMWRITE_PNG_COMPRESSION, 0])


