"""
author: Viktora Dergunova

WHAT IS THIS PROGRAMM FOR ?

This programm lets you compare your camera caliberation images and with different camera models in OpenCV,
to find out the best fit for the data and see if you have enough images for a good caliberation

"""

import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.ticker import SymmetricalLogLocator, MaxNLocator, FormatStrFormatter

# DATA
frameSize = (5168, 3448)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.CALIB_CB_NORMALIZE_IMAGE + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

all_images = glob.glob("./data/test/SD128_35mm/*.JPG")

#get image points once to save memory
def get_image_points(images):
    objpoints = []
    imgpoints = []
    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(np.copy(objp))
            imgpoints.append(corners2)
    return objpoints, imgpoints

objpoints, imgpoints = get_image_points(all_images)


calibration_cache = {}

def calibrate_camera(objpoints, imgpoints, num_params):

    cache_key = (len(objpoints), num_params)
    if cache_key in calibration_cache:
        return calibration_cache[cache_key]
    # sets the modesl 2,3,5,8
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

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frameSize, None, None, flags=flags
    )

    dist = dist.flatten()
    if num_params == 8 and dist.size < 8:
        dist = np.hstack([dist, np.zeros(8 - dist.size)])
    elif dist.size < num_params:
        raise RuntimeError(f"Expected {num_params} distortion parameters, but only got {dist.size}.")

    #print(f"Distortion Coefficients (Params={num_params}): {dist}")

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        imgpoints2 = imgpoints2.reshape(-1, 2).astype(np.float32)
        imgpoints_flat = imgpoints[i].reshape(-1, 2).astype(np.float32)
        error = cv.norm(imgpoints_flat, imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error

    rpe = total_error / len(objpoints)

    #store the rsults in cache to save memory
    calibration_cache[cache_key] = (ret, cameraMatrix, dist, rvecs, tvecs, rpe)
    return calibration_cache[cache_key]

# COLLECTION OF DIST PARAMS 
def parallel_collect_dist_params(images, objpoints, imgpoints, num_params_list, total_images):
    def run_calibration_for_subset(train_size):
        subset_results = {}
        current_train_objpoints = objpoints[:train_size]
        current_train_imgpoints = imgpoints[:train_size]

        for num_params in num_params_list:
            _, _, dist_params, _, _, _ = calibrate_camera(current_train_objpoints, current_train_imgpoints, num_params)
            subset_results[num_params] = dist_params

        return train_size, subset_results

    results = Parallel(n_jobs=-1)(
        delayed(run_calibration_for_subset)(train_size) for train_size in range(1, total_images + 1)
    )
    return results

def collect_dist_params_iterative_parallel(images, objpoints, imgpoints, num_params_list):
    total_train_images = len(images)
    dist_params_dict = {num_params: [] for num_params in num_params_list}

    results = parallel_collect_dist_params(images, objpoints, imgpoints, num_params_list, total_train_images)

    for train_size, dist_results in results:
        for num_params, dist_params in dist_results.items():
            dist_params_dict[num_params].append(dist_params)

    return list(range(1, total_train_images + 1)), dist_params_dict

#RPE + AIC COLLECTION, PLOTTING 
def calculate_aic(rpe, num_params, n):
    k = num_params + 4  #  exclude  fx, fy, cx, cy
    scaled_rpe = rpe * 1000  #scale to get positive number, would not matter if its negative for aic 
    aic = 2 * k + n * np.log(scaled_rpe)
    return aic

def parallel_calibrate_camera_for_rpe(images, objpoints, imgpoints, num_params_list):
    total_images = len(images)

    def calibrate_for_subset(subset_size):
        rpe_dict = {}
        current_objpoints = objpoints[:subset_size]
        current_imgpoints = imgpoints[:subset_size]

        for num_params in num_params_list:
            _, _, _, _, _, rpe = calibrate_camera(current_objpoints, current_imgpoints, num_params)
            rpe_dict[num_params] = rpe
        return subset_size, rpe_dict

    results = Parallel(n_jobs=-1)(
        delayed(calibrate_for_subset)(subset_size) for subset_size in range(1, total_images + 1)
    )
    return results

def collect_and_plot_rpe_parallel(images, objpoints, imgpoints, num_params_list):
    total_images = len(images)
    rpe_dict = {num_params: [] for num_params in num_params_list}
    final_rpes = {}
    aic_values = {}

    results = parallel_calibrate_camera_for_rpe(images, objpoints, imgpoints, num_params_list)

    for subset_size, rpe_values in results:
        for num_params, rpe in rpe_values.items():
            rpe_dict[num_params].append(rpe)

    for num_params in num_params_list:
        final_rpes[num_params] = rpe_dict[num_params][-1]
        aic_values[num_params] = calculate_aic(
            rpe_dict[num_params][-1], num_params, len(all_images) * len(objpoints[0])
        )

    plt.figure(figsize=(10, 6))
    linestyles = ['-', '--', '-.', ':']

    for idx, num_params in enumerate(rpe_dict.keys()):
        linestyle = linestyles[idx % len(linestyles)]
        rpes = rpe_dict[num_params]

        plt.plot(
            range(1, total_images + 1),
            rpes,
            linestyle=linestyle,
            label=f'RPE mit {num_params} Parameter'
        )

    final_legend_entries = [
        f'Parameter={num_params}: Finaler RPE={final_rpes[num_params]:.5f}, AIC={aic_values[num_params]:.2f}'
        for num_params in num_params_list
    ]
    plt.legend(final_legend_entries, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

    plt.xlabel('Anzahl der Bilder')
    plt.ylabel('Reproduktionsfehler (RPE)')
    plt.title('RPE über verschiedene Kameraparameter')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


# for log scale
def calculate_linthresh(min_val, max_val):
    range_val = max_val - min_val
    return max(1e-2, range_val * 0.5)

#PLOTTING OF DIST PARAMS 

def plot_distortion_parameters_parallel(num_images_list, dist_params_dict, dist_labels, num_params_list):
    for i, label in enumerate(dist_labels):
        plt.figure(figsize=(10, 6))
        
        linestyles = ['-', '--', '-.', ':']
        legend_entries = []

        min_val = float('inf')
        max_val = float('-inf')
        
        for j, num_params in enumerate(num_params_list):
            if i >= len(dist_params_dict[num_params][0]):
                continue
                
            initial_values = np.array([dist[i] for dist in dist_params_dict[num_params]])

            min_val = min(min_val, np.min(initial_values))
            max_val = max(max_val, np.max(initial_values))

            if not np.all(initial_values == 0):
                line, = plt.plot(
                    num_images_list,
                    initial_values,
                    linestyle=linestyles[j % len(linestyles)],
                    label=f'{label} (Parameter={num_params})'
                )
                final_value = initial_values[-1]
                legend_entries.append(
                    (line, f'{label} (Parameter={num_params}): {final_value:.5f}')
                )
        
        if legend_entries:
            plt.legend(
                [entry[0] for entry in legend_entries],
                [entry[1] for entry in legend_entries],
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                frameon=False
            )
            plt.xlabel('Anzahl der Bilder')
            #Y SCALE
            # take symlog if detection of negative values, else take log 
            linthresh = calculate_linthresh(min_val, max_val)
            plt.yscale('symlog', linthresh=linthresh)
            plt.gca().yaxis.set_major_locator(SymmetricalLogLocator(linthresh=linthresh, base=10.0))

            range_val = max_val - min_val
            buffer = range_val * 0.05 # add some buffer so plots are not cramped together

            lower_limit = min_val - buffer
            upper_limit = max_val + buffer

            plt.ylim(lower_limit, upper_limit)

            #log 
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
            
            plt.ylabel(f'{label} Verzerrungsparameter')
            plt.title(f'Verzerrungsparameter {label} über verschiedenen Kameramodelle')
            plt.grid(True, which="both", ls="--")
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.show()

#PLOOTING FOR COMBINED RADIAL DIST
def plot_combined_radial_distortion_all_models(all_dist_params_dict, num_params_list):
    r = np.linspace(0, 1, 100)
    plt.figure(figsize=(12, 8))
    
    linestyles = ['-', '--', '-.', ':']  

    for idx, num_params in enumerate(num_params_list):
        k1_actual = all_dist_params_dict[num_params][-1][0]
        k2_actual = all_dist_params_dict[num_params][-1][1] if len(all_dist_params_dict[num_params][-1]) > 1 else 0
        k3_actual = all_dist_params_dict[num_params][-1][4] if len(all_dist_params_dict[num_params][-1]) > 2 else 0
        k3_actual
        
        #calculate radial combined dist
        distortion_actual = (k1_actual * r**2 + k2_actual * r**4 + k3_actual * r**6 )

        #plot my models
        plt.plot(r, distortion_actual, label=f'Parameter={num_params}', 
                 linestyle=linestyles[idx % len(linestyles)])

    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Normalisierte radiale Distanz (r)')
    plt.ylabel('Verzerrung')
    plt.title('Kombinierte radiale Verzerrungskurve über verschiedene Kameramodelle')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    plt.grid(True)
    plt.show()


# RUNNING THE ANALYSIS
def run_dis_plot(num_params_list, all_images, objpoints, imgpoints, dist_labels):

    print("Collecting distortion parameters...")
    num_images_list, all_dist_params_dict = collect_dist_params_iterative_parallel(
        all_images, objpoints, imgpoints, num_params_list
    )

    #print("Plotting distortion parameters...")
    plot_distortion_parameters_parallel(num_images_list, all_dist_params_dict, dist_labels, num_params_list)

    return num_images_list, all_dist_params_dict

def run_rpe(num_params_list, all_images, objpoints, imgpoints):

    #print("Plotting RPE ...")
    collect_and_plot_rpe_parallel(all_images, objpoints, imgpoints, num_params_list)

def run_combined_plot(all_dist_params_dict, num_params_list):

    #print("Plotting combined radial distortion curves...")
    plot_combined_radial_distortion_all_models(all_dist_params_dict, num_params_list)


if __name__ == "__main__":

    objpoints, imgpoints = get_image_points(all_images)
    
    num_params_list = [2, 3, 5,8]
    all_dist_labels = ["k1", "k2", "p1", "p2", "k3"]
    
    num_images_list, all_dist_params_dict = run_dis_plot(num_params_list, all_images, objpoints, imgpoints, all_dist_labels)
    #un_rpe(num_params_list, all_images, objpoints, imgpoints)
    run_combined_plot(all_dist_params_dict, num_params_list)
