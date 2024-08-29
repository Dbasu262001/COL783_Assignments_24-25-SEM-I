# %%
import numpy as np
import pandas as pd
import cv2
import os
image2 = cv2.imread("data/memorial.hdr",cv2.IMREAD_ANYDEPTH)


# %% [markdown]
# Q2 A 

# %%
#**************Q2 A***************
gray_image2 = np.mean(image2,axis=2)
print("gray_image shape",gray_image2.shape)
np.max(gray_image2)
print("******** Q2 a ************")
max_intensity = np.max(gray_image2)
min_intensity = np.min(gray_image2)
contrast_ratio = (max_intensity)/(min_intensity)
print("maximum intensity = ",max_intensity)
print("minimum intensity = ", min_intensity)
print("contrast ratio = ", contrast_ratio)

# %%


# %%
c1 = 1/min_intensity
scaled_image_c1 = gray_image2 * c1
diff = np.max(scaled_image_c1) - np.min(scaled_image_c1)
scaled_image_c1 = (scaled_image_c1 - np.min(scaled_image_c1)) *(255/diff)
np.max(scaled_image_c1)

# %%
cv2.imwrite("output_images/Q2/Q2_a_min_intensity_1.png",scaled_image_c1,[cv2.IMWRITE_PNG_COMPRESSION,0])


# %%
# cv2.imshow('minimum intensity 1 Image', scaled_image_c1)  # Multiply by 255 to visualize
# cv2.waitKey(8000)
# cv2.destroyAllWindows()

# %%
c2 = 255/max_intensity
scaled_image_c2 = gray_image2*c2
# np.min(scaled_image_c2)

# %%
cv2.imwrite("output_images/Q2/Q2_a_max_intensity_255.png",scaled_image_c2,[cv2.IMWRITE_PNG_COMPRESSION,0])


# %%
# cv2.imshow('maximum intensity is 255 Image', scaled_image_c2)  # Multiply by 255 to visualize
# cv2.waitKey(8000)
# cv2.destroyAllWindows()

# %% [markdown]
# Q2 B

# %%
#******************Q2 B*****************
#Log transfprmation

log_image = np.log(gray_image2) # image after log operation
#scaling a*log(r) + b
log_image_t = (255/(np.max(log_image)-np.min(log_image)))*log_image -(255/(np.max(log_image)-np.min(log_image)))*np.min(log_image)
#clipping
log_transformed_image = (((log_image_t - np.min(log_image_t))/(np.max(log_image_t)-np.min(log_image_t)))*255).astype(np.uint8)
cv2.imwrite("output_images/Q2/Q2_b_log_tranformed.png",log_transformed_image,[cv2.IMWRITE_PNG_COMPRESSION,0])


# %%

# cv2.imshow('log scaling Image', log_transformed_image)  # Multiply by 255 to visualize
# cv2.waitKey(8000)
# cv2.destroyAllWindows()

# %% [markdown]
# Q2 C

# %%
#************* Q2 C***********************
#************(exponent of log transformed)
log_image =np.log(gray_image2) # image after using log operation
a = np.log(255)/(np.max(log_image) - np.min(log_image))
b = -a * np.min(log_image)
linear_transform = a*log_image +b
exponent_image = np.exp(linear_transform).astype(np.uint8)
print("minimum value of c exponent image =",np.min(exponent_image))
cv2.imwrite("output_images/Q2/Q2_c_exp_after_log.png",exponent_image,[cv2.IMWRITE_PNG_COMPRESSION,0])


# %%
# cv2.imshow('exponent of log scaling Image', exponent_image)  # Multiply by 255 to visualize
# cv2.waitKey(8000)
# cv2.destroyAllWindows()

# %% [markdown]
# Q2 D

# %%
#**************Q2 D***********
log_image = np.log(gray_image2)
def histogram_equalization(image:np.array):
    flattened_image = image.flatten().astype(np.int16)
    histogram = np.bincount(flattened_image,minlength=256)
    cdf = histogram.cumsum()
    normalized_cdf = 255*((cdf - cdf.min())/(cdf.max() - cdf.min()))
    equalized_image = normalized_cdf[flattened_image]
    equalized_image = equalized_image.reshape(image.shape).astype(np.uint8)
    return equalized_image
gray_image_histo_equalized=histogram_equalization(gray_image2)


# %%
cv2.imwrite("output_images/Q2/Q2_d_histo_eq_gray.png",exponent_image,[cv2.IMWRITE_PNG_COMPRESSION,0])

# %%
# cv2.imshow('histogram_equalized Image without log', gray_image_histo_equalized)  # Multiply by 255 to visualize
# cv2.waitKey(8000)
# cv2.destroyAllWindows()

# %%
log_image_eq = log_image_t.astype(np.int16) #image after log transform
log_equailized_image = histogram_equalization(log_image_eq)
cv2.imwrite("output_images/Q2/Q2_d_histo_eq_with_log.png",log_equailized_image,[cv2.IMWRITE_PNG_COMPRESSION,0])


# %%
# cv2.imshow('histogram_equalized Image after log', log_equailized_image)  # Multiply by 255 to visualize
# cv2.waitKey(8000)
# cv2.destroyAllWindows()

# %%
#******************Q2 E**************
my_image = cv2.imread("data/palace.jpg",cv2.IMREAD_ANYDEPTH)

# %%
def histogram_matching(source:np.array,target:np.array):
    source_histo = np.bincount(source.flatten().astype(np.int16))
    target_histo = np.bincount(target.flatten().astype(np.int16))
    source_cdf = source_histo.cumsum()
    source_cdf = source_cdf /source_cdf[-1]
    target_cdf = target_histo.cumsum()
    target_cdf = target_cdf /target_cdf[-1]
    matching_table = np.zeros(256,dtype=np.uint8)
    t=0
    for i in range(256):
        while (t < 256 and target_cdf[t] < source_cdf[i] ):
            t+=1
        matching_table[i] = t
    matched_image = matching_table[source.flatten().astype(np.int16)]
    return matched_image.reshape(source.shape)



# %%
histogram_match_image =histogram_matching(log_equailized_image,my_image)

# %%
cv2.imwrite("output_images/Q2/Q2_e_original.png",my_image.astype(np.uint8),[cv2.IMWRITE_PNG_COMPRESSION,0])

cv2.imwrite("output_images/Q2/Q2_e_histogram_matching.png",histogram_match_image,[cv2.IMWRITE_PNG_COMPRESSION,0])


# %%
# cv2.imshow('histogram matching Image', histogram_match_image)  # Multiply by 255 to visualize
# cv2.waitKey(8000)
# cv2.destroyAllWindows()


