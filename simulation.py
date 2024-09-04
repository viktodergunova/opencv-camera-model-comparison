"""
author: Viktora Dergunova

WHAT IS THIS PROGRAMM FOR?

This program lets you simulate the effect of a handheld camera
"""

import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from matplotlib.ticker import SymmetricalLogLocator, MaxNLocator, FormatStrFormatter


# DATA
frameSize = (5168, 3448)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

all_images = glob.glob("./data/SD128_35mm/*.JPG")

def get_image_points(images, apply_handheld=True, rotation_angle_range=(-5, 5), max_translation_range=(0, 30), probability=0.6):
    objpoints = []
    imgpoints = []
    for image in images:
        img = cv.imread(image)

        if apply_handheld:
            img = apply_handheld_simulation(img, rotation_angle_range, max_translation_range, probability)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(np.copy(objp))
            imgpoints.append(corners2)
    return objpoints, imgpoints


# HANDHELD SIMULATION
def apply_handheld_simulation(image, rotation_angle_range, max_translation_range, probability):
    if random.random() < probability:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # randomly selects rotation angle and translation values within the provided ranges
        rotation_angle = random.uniform(rotation_angle_range[0], rotation_angle_range[1])
        max_translation = random.uniform(max_translation_range[0], max_translation_range[1])

        rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1.0)

        translation_x = np.random.uniform(-max_translation, max_translation)
        translation_y = np.random.uniform(-max_translation, max_translation)
        rotation_matrix[0, 2] += translation_x
        rotation_matrix[1, 2] += translation_y

        simulated_image = cv.warpAffine(image, rotation_matrix, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        return simulated_image
    else:
        # returns normal image if not affected
        return image


def simulate_dist_params_with_handheld(objpoints, imgpoints, num_params, num_images_list, rotation_angle_range, max_translation_range, probability):
    def process_image_subset(num_images):
        noisy_imgpoints = []
        for img_path, pts in zip(all_images[:num_images], imgpoints[:num_images]):
            image = cv.imread(img_path)

        
            if random.random() < probability:
                image = apply_handheld_simulation(image, rotation_angle_range, max_translation_range, probability)

        
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
                if ret:
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    noisy_imgpoints.append(corners2)
                else:
                    noisy_imgpoints.append(pts) 
            else:
                noisy_imgpoints.append(pts)

        _, _, dist_coeffs, _, _, rpe = calibrate_camera(objpoints[:num_images], noisy_imgpoints, num_params)
        return dist_coeffs.flatten(), rpe

    results = Parallel(n_jobs=-1)(delayed(process_image_subset)(num_images) for num_images in num_images_list)
    
    dist_simulations, rpe_simulations = zip(*results)
    return np.array(dist_simulations), np.array(rpe_simulations)


def calibrate_camera(objpoints, imgpoints, num_params):
    if num_params == 2:
        flags = cv.CALIB_FIX_K3 | cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6
    elif num_params == 3:
        flags = cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6
    elif num_params == 5:
        flags = cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6
    elif num_params == 8:
        flags = 0
    else:
        raise ValueError("Number of parameters must be 2, 3, 5, or 8.")
    
    ret, cameraMatrix, dist, rvecs, tvecs, stdDevsIntrinsics, stdDevsExtrinsics, perViewErrors = cv.calibrateCameraExtended(
        objpoints, imgpoints, frameSize, None, None, flags=flags)
    
    dist = dist.flatten()
    if num_params == 8:
        if dist.size < 8:
            dist = np.hstack([dist, np.zeros(8 - dist.size)])
    elif dist.size < num_params:
        raise RuntimeError(f"Expected {num_params} distortion parameters, but only got {dist.size}.")

    print(f"Distortion Coefficients: {dist}")

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        imgpoints2 = imgpoints2.reshape(-1, 2).astype(np.float32)
        imgpoints_flat = imgpoints[i].reshape(-1, 2).astype(np.float32)
        error = cv.norm(imgpoints_flat, imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
        
    rpe = total_error / len(objpoints)
    return ret, cameraMatrix, dist.flatten(), rvecs, tvecs, rpe

#PLOT RADIAL DIST TOGETHER
def plot_combined_radial_distortion_curve(k1_actual, k2_actual, k3_actual, k1_sim, k2_sim, k3_sim):
    r = np.linspace(0, 1, 100)
    distortion_actual = (k1_actual * r**2 + k2_actual * r**4 + k3_actual * r**6)
    distortion_sim = (k1_sim * r**2 + k2_sim * r**4 + k3_sim * r**6)

    plt.figure(figsize=(8, 5))
    plt.plot(r, distortion_actual, label="Initial Kaliberation: Radiale Verzerrung", color='blue')
    plt.plot(r, distortion_sim, label="Simulation Kaliberation: Radiale Verzerrung", linestyle='--', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Normalisierte radiale Distanz (r)')
    plt.ylabel('Verzerrung')
    plt.title('Kombinierte radiale Verzerrungskurve über verschiedene Kameramodelle')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_linthresh(min_val, max_val):
    range_val = max_val - min_val
    return max(1e-2, range_val * 0.5)

def plot_distortion_parameters_with_std_dev(num_images_list, dist_params_list, dist_labels, sim_distributions, sim_rpe, final_actual_rpe, simulation_type):
    for i, label in enumerate(dist_labels):
        plt.figure(figsize=(10, 6))

        initial_values = np.array([dist[i] for dist in dist_params_list])
        sim_values = np.array([sim_distributions[j][i] for j in range(len(num_images_list))])

        plt.plot(num_images_list, initial_values, '-', color='blue', label=f'{label} Kaliberation')
        plt.plot(num_images_list, sim_values, 'r--', label=f'{label} Simulation', alpha=0.7)

        final_initial_value = initial_values[-1]
        final_simulation_value = sim_values[-1]
        plt.annotate(f'Initial {label}: {final_initial_value:.5f}', xy=(num_images_list[-1], final_initial_value),
                     xytext=(-50, 20), textcoords='offset points', fontsize=9, color='blue')
                    
        plt.annotate(f'Sim {label}: {final_simulation_value:.5f}', xy=(num_images_list[-1], final_simulation_value),
                     xytext=(-50, -30), textcoords='offset points', fontsize=9, color='red')

        plt.xlabel('Anzahl der Bilder')
        plt.ylabel(f'{label} Verzerrung')
        plt.title(f'Verzerrungsparameter {label} {simulation_type} {num_params} Parameter Kameramodell')
        
    
        min_val = min(np.min(initial_values), np.min(sim_values))
        max_val = max(np.max(initial_values), np.max(sim_values))
        linthresh = calculate_linthresh(min_val, max_val)
        plt.yscale('symlog', linthresh=linthresh)
        plt.gca().yaxis.set_major_locator(SymmetricalLogLocator(linthresh=linthresh, base=10.0))

        range_val = max_val - min_val
        buffer = range_val * 0.05  
        lower_limit = min_val - buffer
        upper_limit = max_val + buffer

        plt.ylim(lower_limit, upper_limit)

        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.grid(True)
        plt.show()

        # PLOT RADIAL DIST AND TANGENTIAL DIST
        if label == 'k1':
            final_calib_dist_coefficients = dist_params_list[-1]  
            final_sim_dist_coefficients = sim_distributions[-1] 

            # final dist calib
            k1_actual = final_calib_dist_coefficients[0] if len(final_calib_dist_coefficients) > 0 else 0
            k2_actual = final_calib_dist_coefficients[1] if len(final_calib_dist_coefficients) > 1 else 0
            k3_actual = final_calib_dist_coefficients[4] if len(final_calib_dist_coefficients) > 4 else 0

            print("Finaler Kalib K1:", k1_actual, "K2:", k2_actual, "K3:", k3_actual)

            # final dist sim
            k1_sim = final_sim_dist_coefficients[0] if len(final_sim_dist_coefficients) > 0 else 0
            k2_sim = final_sim_dist_coefficients[1] if len(final_sim_dist_coefficients) > 1 else 0
            k3_sim = final_sim_dist_coefficients[4] if len(final_sim_dist_coefficients) > 4 else 0

            print("Finaler Sim K1:", k1_sim, "K2:", k2_sim, "K3:", k3_sim)

            # Plot the combined radial distortion curve
            plot_combined_radial_distortion_curve(
                k1_actual, k2_actual, k3_actual,
                k1_sim, k2_sim, k3_sim
            )

# ITERATIVE DIST PARAMS WITH PARALLELIZATION
def collect_dist_params_iterative(images, objpoints, imgpoints, num_params):
    def process_subset(train_size):
        current_train_imgs = images[:train_size]
        current_train_objpoints = objpoints[:train_size]
        current_train_imgpoints = imgpoints[:train_size]

        _, _, dist_params, _, _, rpe = calibrate_camera(current_train_objpoints, current_train_imgpoints, num_params)

        return train_size, dist_params.flatten(), rpe

    results = Parallel(n_jobs=-1)(delayed(process_subset)(train_size) for train_size in range(1, len(images) + 1))
    
    num_images_list, dist_params_list, actual_rpe_list = zip(*results)
    
    return list(num_images_list), list(dist_params_list), list(actual_rpe_list)

### RUN ###
num_params_list = [5]
all_dist_labels = ["k1", "k2", "p1", "p2","k3"]
n_points_per_image = chessboardSize[0] * chessboardSize[1]
n_total_points = len(all_images) * n_points_per_image

all_actual_rpes = {}
all_sim_rpes = {}
final_aic_values = {}

# SIMULATION PARAMETERS
rotation_angle_range = (-5, 5)  # degrees
max_translation_range = (0, 20)  # max pixel translation in x and y direction
probability = 0.6

for num_params in num_params_list:
    print(f"Testing with num_params={num_params}")
    
    objpoints, imgpoints = get_image_points(all_images)
    # INITIAL CALIBRATION
    num_images_list, dist_params_list, actual_rpe_list = collect_dist_params_iterative(all_images, objpoints, imgpoints, num_params)
    final_actual_rpe = actual_rpe_list[-1]

    all_actual_rpes[num_params] = actual_rpe_list

    current_labels = all_dist_labels[:num_params]

    # HANDHELD SIMULATION
    sim_distributions_h, sim_rpe_h = simulate_dist_params_with_handheld(
        objpoints, imgpoints, num_params, num_images_list, rotation_angle_range, max_translation_range, probability)
    all_sim_rpes[num_params] = sim_rpe_h

    # PLOT HANDHELD SIMULATION
    plot_distortion_parameters_with_std_dev(
        num_images_list, dist_params_list, current_labels, sim_distributions_h, sim_rpe_h, 
        final_actual_rpe, simulation_type="Handkamera Simulation"
    )

    final_sim_rpe = sim_rpe_h[-1]


plt.figure(figsize=(10, 6))

for num_params in num_params_list:
    plt.plot(num_images_list, all_actual_rpes[num_params], '-', label=f'Initiale Kaliberation RPE (Parameter={num_params})')
    plt.plot(num_images_list, all_sim_rpes[num_params], '--', label=f'Simulation RPE (Parameter={num_params})')

final_legend_entries = [f'Parameter={num_params}: Initiale Kaliberation RPE={all_actual_rpes[num_params][-1]:.5f}, Finaler Simulation RPE={all_sim_rpes[num_params][-1]:.5f}' for num_params in num_params_list]
plt.legend(final_legend_entries, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.xlabel('Anzahl der Bilder')
plt.ylabel('Reproduktionsfehler (RPE)')
plt.title('RPE über verschiedene Kameraparameter')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.grid(True)
plt.show()
