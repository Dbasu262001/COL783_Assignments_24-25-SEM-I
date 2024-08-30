# %%
import numpy as np
import pandas as pd
import cv2
import os
image4 = cv2.imread("data/vinesunset.hdr",cv2.IMREAD_ANYDEPTH)
import matplotlib.pyplot as plt


# %%
#************ Q4 a************8
# Converting hsi image to rgb image
def hsi_to_rgb_value(h, s, i):
    h = h * 360
    if(h >= 240):
        Hs = h-240
    elif(h >= 120):
        Hs = h-120
    else:
        Hs = h
    a = i * (1 - s)
    b = i * (1 + s * np.cos(np.radians(Hs)) / np.cos(np.radians(60 - Hs)))
    c = 3*i - (a+b)
    if h < 120:
        b,r,g = a,b,c
    elif h < 240:
        r,g,b = a,b,c
    else:
        g,b,r = a,b,c
    return np.clip(np.array([r, g, b]), 0, 1)

# Creating an hsi image 
def create_hsi_image(image_size=256,constant='hue',val = 0.5):
    image = np.zeros((image_size,image_size, 3), dtype=np.float32)
    m = image_size-1
    for y in range(m):
        for x in range(m):
            if constant == 'hue':
                s = x / m
                i = y / m
                rgb = hsi_to_rgb_value(val, s, i)
            elif constant == 'saturation':
                h = x / m
                i = y / m
                rgb = hsi_to_rgb_value(h, val, i)
            elif constant == 'intensity':
                h = x / m
                s = y / m
                rgb = hsi_to_rgb_value(h, s, val)
            if np.any(rgb < 0) or np.any(rgb > 1):
                rgb = [1, 0, 0]  # Error color (red)
            image[y, x] = rgb
    return (image * 255).astype(np.uint8)

# Create images
hue_constant_image = create_hsi_image(image_size=400,constant='hue',val=0.9)
saturation_constant_image = create_hsi_image(image_size=400,constant='saturation',val=0.7)
intensity_constant_image = create_hsi_image(image_size=400,constant='intensity',val=0.4)

# Save images
cv2.imwrite('output_images/Q4/Q4_a_hue_constant.png', hue_constant_image)
cv2.imwrite('output_images/Q4/Q4_a_saturation_constant.png', saturation_constant_image)
cv2.imwrite('output_images/Q4/Q4_a_intensity_constant.png', intensity_constant_image)


# %%
######********** Q4 c******************

# %%
#### RGB  image to HSI
def rgb_to_hsi(image:np.array):
    normalised_img = image /255.0
    I = np.sum(normalised_img,axis=2)/3.0
    r = normalised_img[:,:,0]
    g = normalised_img[:,:,1]
    b = normalised_img[:,:,2]
    S = 1 - (3*np.minimum(r,np.minimum(g,b))/(r+g+b +1e-9))
    hue_angle = np.arccos(0.5 * (2*r - g - b) / (np.sqrt( np.square(r-g) + (r-b)*(g-b) +1e-9 )))
    H = 0.5*np.where(b<=g,hue_angle,2*np.pi - hue_angle)/np.pi
    hsi = np.stack([H,S,I],axis=-1)
    return hsi

## #******** HSI image to hsi
def hsi_to_rgb_image(image:np.array):
    rgb_image = np.zeros(shape=image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            rgb = hsi_to_rgb_value(image[x,y,0],image[x,y,1],image[x,y,2])
            if np.any(rgb < 0) or np.any(rgb > 1):
                rgb = [1, 0, 0]  # Error color (red)
            image[x, y] = rgb
    return (image * 255.0)

# %%
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
    padded_image = np.pad(image1,padding_size,mode='reflect')
    image = np.zeros(image1.shape)
    m,n = image.shape[0],image.shape[1]
    for i in range(m):
        for j in range(n):
            f_q = padded_image[i:i+size,j:j+size]
            fp_fq = padded_image[i + padding_size, j + padding_size] - padded_image[i:i+size,j:j+size]
            gauss_range = np.exp(-0.5*np.square(fp_fq)/np.square(sigma_r))
            weights = gauss_range * gauss_spatial
            
            image[i,j] =np.sum(f_q * weights /np.sum(weights))
    return image 



# %%
hsi_image = rgb_to_hsi(image4)
intensity = (hsi_image[:, :, 2] * 255)  # Convert Intensity to 8-bit
filtered_intensity = bilater_filter(intensity,2,2)  # Apply bilateral filter
hsi_image[:, :, 2] = filtered_intensity/255
rgb_from_hsi = hsi_to_rgb_image(hsi_image)

# %%
# cv2.imshow('Q1(a) sep background & foreground', rgb_from_hsi.astype(np.uint8))
# cv2.waitKey(7500)
# cv2.destroyAllWindows()

# %%
#Q4 C***********
# Q4 C i
hsi_img_1 = rgb_to_hsi(image=image4)
I = hsi_img_1[:,:,2]*255
I_reduced = bilater_filter(I,2,2)
hsi_reduced = np.stack([hsi_img_1[:,:,0],hsi_img_1[:,:,1],I_reduced.astype(float)/255.0],axis=-1)


# %%
I_reduced_rgb = hsi_to_rgb_image(hsi_reduced)

# %%
cv2.imwrite('output_images/Q4/Q4_c_Intensity_reduced.png', I_reduced_rgb)

# %%
# R reduced
R_reduced = bilater_filter(image4[:,:,0],2,2)
R_reduced_scale = (255*(R_reduced - np.min(R_reduced)) / (np.max(R_reduced) - np.min(R_reduced)))
R_reduced_image = np.stack([R_reduced_scale,image4[:,:,1],image4[:,:,2]],axis=-1)
cv2.imwrite('output_images/Q4/Q4_c_R_reduced.png', R_reduced_image.astype(np.uint8))

# %%
G_reduced = bilater_filter(image4[:,:,1],2,2)
G_reduced_scale = (255*(G_reduced - np.min(G_reduced)) / (np.max(G_reduced) - np.min(G_reduced)))

G_reduced_image = np.stack([image4[:,:,0],G_reduced_scale,image4[:,:,2]],axis=-1)
cv2.imwrite('output_images/Q4/Q4_c_G_reduced.png', G_reduced_image.astype(np.uint8))

# %%
B_reduced = bilater_filter(image4[:,:,2],2,2)
B_reduced_scale = (255*(B_reduced - np.min(B_reduced)) / (np.max(B_reduced) - np.min(B_reduced)))

B_reduced_image = np.stack([image4[:,:,0],image4[:,:,1],B_reduced_scale],axis=-1)
cv2.imwrite('output_images/Q4/Q4_c_B_reduced.png', B_reduced_image.astype(np.uint8))

# %%


# %%
#***********
#************Q4 d************
def gamma_correction(image:np.array,gamma:float):
    image = image/255.0
    gamma_corrected_image = image**gamma
    gamma_corrected_image = 255*gamma_corrected_image

    scaled_image = 255*((gamma_corrected_image - np.min(gamma_corrected_image))/(np.max(gamma_corrected_image) - np.min(gamma_corrected_image)))
    return scaled_image.astype(np.uint8)




# %%
I_reduced_gamma = gamma_correction(I_reduced_rgb,(1/2.2))
cv2.imwrite('output_images/Q4/Q4_d_I_reduced_gamma.png', I_reduced_gamma.astype(np.uint8))

# %%
R_reduced_gamma = gamma_correction(R_reduced_image,(1/2.2))
cv2.imwrite('output_images/Q4/Q4_d_R_reduced_gamma.png', R_reduced_gamma.astype(np.uint8))

# %%
G_reduced_gamma = gamma_correction(G_reduced_image,(1/2.2))
cv2.imwrite('output_images/Q4/Q4_d_G_reduced_gamma.png', G_reduced_gamma.astype(np.uint8))

# %%
B_reduced_gamma = gamma_correction(B_reduced_image,(1/2.2))
cv2.imwrite('output_images/Q4/Q4_d_B_reduced_gamma.png', B_reduced_gamma.astype(np.uint8))


