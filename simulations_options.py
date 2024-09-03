
"""
author: Viktora Dergunova

"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

image_path = "./data/test/SD128_35mm/DSC00050.JPG"  
image = cv.imread(image_path)

def apply_autofocus_simulation(image, blur_level):
    ksize = blur_level * 2 + 1  
    if ksize > 1:
        blurred_image = cv.GaussianBlur(image, (ksize, ksize), 0)
    else:
        blurred_image = image  
    return blurred_image


def apply_aperture_blur_simulation(image, blur_level):
    depth_of_field_blur = cv.GaussianBlur(image, (blur_level, blur_level), blur_level)
    return depth_of_field_blur

def apply_motion_blur_simulation(image, blur_length_range, blur_angle_range, blur_probability):
    if random.random() < blur_probability:
        blur_length = random.randint(blur_length_range[0], blur_length_range[1])
        blur_angle = random.uniform(blur_angle_range[0], blur_angle_range[1])
        
        ksize = blur_length
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize-1)/2), :] = np.ones(ksize)
        kernel = cv.warpAffine(kernel, cv.getRotationMatrix2D((ksize/2 - 0.5, ksize/2 - 0.5), blur_angle, 1), (ksize, ksize))
        kernel = kernel / np.sum(kernel)
        motion_blurred_image = cv.filter2D(image, -1, kernel)
        return motion_blurred_image
    else:
        return image  

def apply_lighting_simulation_with_shot_noise(image, brightness_factor, noise_scale_factor):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    
    v = cv.add(v, brightness_factor)
    v = np.clip(v, 0, 255)
    
    v = np.random.poisson(v.astype(float) * noise_scale_factor) / noise_scale_factor
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    final_hsv = cv.merge((h, s, v))
    return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)

def apply_handheld_simulation(image, rotation_angle, max_translation):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1.0)

    translation_x = np.random.uniform(-max_translation, max_translation)
    translation_y = np.random.uniform(-max_translation, max_translation)
    rotation_matrix[0, 2] += translation_x
    rotation_matrix[1, 2] += translation_y

    simulated_image = cv.warpAffine(image, rotation_matrix, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    return simulated_image

# PARAMETERS
blur_level_autofocus = 19  # blur level for autofocus simulation, must be positive, odd
blur_level_aperture = 99  # blur level for aperture, positive, odd
blur_length_range_motion = (60, 70)  # range of motion blur lengths
blur_angle_range_motion = (0,90)  # range of motion blur angles (degrees)
blur_probability = 1.0  # Probability of applying motion blur to any image
brightness_factor_low = -19  # -x - x brightness adjustment
brightness_factor_bright = 40 
noise_scale_factor = 0.1  # shot noise
rotation_angle = 5  # degrees
max_translation = 30  # max pixel translation in x and y direction


autofocus_simulated_image = apply_autofocus_simulation(image, blur_level_autofocus)
aperture_blur_simulated_image = apply_aperture_blur_simulation(image, blur_level_aperture)
motion_blurred_image = apply_motion_blur_simulation(image, blur_length_range_motion, blur_angle_range_motion, blur_probability)
lighting_shot_noise_simulated_image = apply_lighting_simulation_with_shot_noise(image, brightness_factor_low, noise_scale_factor)
lighting_shot_noise_simulated_image_2 = apply_lighting_simulation_with_shot_noise(image, brightness_factor_bright, noise_scale_factor)
handheld_simulated_image = apply_handheld_simulation(image, rotation_angle, max_translation)

fig, axes = plt.subplots(1, 2, figsize=(80, 10))

#ORIGINAL
axes[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis('off')
""" 
# AUTOFOCUS SIM
axes[1].imshow(cv.cvtColor(autofocus_simulated_image, cv.COLOR_BGR2RGB))
axes[1].set_title("Autofocus Simulation")
axes[1].axis('off')

# APERTURE SIM
axes[2].imshow(cv.cvtColor(aperture_blur_simulated_image, cv.COLOR_BGR2RGB))
axes[2].set_title("Aperture Simulation")
axes[2].axis('off')

# MOTION BLUR SIM
axes[2].imshow(cv.cvtColor(motion_blurred_image, cv.COLOR_BGR2RGB))
axes[2].set_title("Motion Simulation")
axes[2].axis('off')

# LIGHTENING SIM 
axes[3].imshow(cv.cvtColor(lighting_shot_noise_simulated_image, cv.COLOR_BGR2RGB))
axes[3].set_title("Lighting low")
axes[3].axis('off')

axes[4].imshow(cv.cvtColor(lighting_shot_noise_simulated_image_2, cv.COLOR_BGR2RGB))
axes[4].set_title("Lighting bright")
axes[4].axis('off')
 """
# WITHOUT STATIV SIM (ROATATION + TRANSLATION)
axes[1].imshow(cv.cvtColor(handheld_simulated_image, cv.COLOR_BGR2RGB))
axes[1].set_title("Handheld Simulation")
axes[1].axis('off')

plt.show()
