import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

# DATA
frameSize = (5168, 3448)
size_of_chessboard_squares_mm = 10
chessboardSize = (17, 28)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

all_images = glob.glob("./data/SD128_35mm/*.JPG")

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

    print(f"Distortion Coefficients (Params={num_params}): {dist}")

    if num_params > 0:
        dist_params_uncertainties = stdDevsIntrinsics[4:4 + num_params] # Extract uncertainties 
    else:
        dist_params_uncertainties = np.array([])

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        imgpoints2 = imgpoints2.reshape(-1, 2).astype(np.float32)
        imgpoints_flat = imgpoints[i].reshape(-1, 2).astype(np.float32)
        error = cv.norm(imgpoints_flat, imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
        
    rpe = total_error / len(objpoints)
    return ret, cameraMatrix, dist.flatten(), rvecs, tvecs, rpe, dist_params_uncertainties

def plot_distortion_parameters_with_std_dev(num_images_list, dist_params_list, dist_labels, num_params_list, uncertainties_list=None):
    if uncertainties_list is not None:
        uncertainties_list = [np.array(uncertainties) for uncertainties in uncertainties_list]
    
    for i, label in enumerate(dist_labels):
        plt.figure(figsize=(10, 6))
        
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 'x', 's', 'd']
        
        valid_model_found = False  # 

        for j, num_params in enumerate(num_params_list):
        
            if i >= len(dist_params_list[j][0]):  
                continue 

            initial_values = np.array([dist[i] for dist in dist_params_list[j]])
            
            if not np.all(initial_values == 0): 
                plt.plot(num_images_list, initial_values, 
                             linestyle=linestyles[j % len(linestyles)], marker=markers[j % len(markers)], 
                             label=f'{label} Calibration (Params={num_params})')

                if uncertainties_list is not None and uncertainties_list[j].shape[1] > i: 
                    lower_bound = initial_values - uncertainties_list[j][:, i]
                    upper_bound = initial_values + uncertainties_list[j][:, i]
                    plt.fill_between(num_images_list, lower_bound, upper_bound, alpha=0.2, label=f'{label} Uncertainty (Params={num_params})')
                
                valid_model_found = True

        
        if valid_model_found: 
            plt.yscale('symlog', linthresh=1e-3) 
            plt.xlabel('Number of Images')
            plt.ylabel(f'{label} Value')
            plt.title(f'Distortion Parameter {label} Variation Across Different Camera Models')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.grid(True, which="both", ls="--")
            plt.show()




# ITERATIVE DIST PARAMS
def collect_dist_params_iterative(images, objpoints, imgpoints, num_params):
    total_train_images = len(images)
    num_images_list = []
    dist_params_list = []
    uncertainties_list = []

    for train_size in range(1, total_train_images + 1):
        current_train_imgs = images[:train_size]
        current_train_objpoints = objpoints[:train_size]
        current_train_imgpoints = imgpoints[:train_size]
        num_images_list.append(len(current_train_imgs))

        _, _, dist_params, _, _, _, uncertainties = calibrate_camera(current_train_objpoints, current_train_imgpoints, num_params)

        dist_params_list.append(dist_params.flatten())
        uncertainties_list.append(uncertainties.flatten() if uncertainties.size > 0 else np.zeros_like(dist_params.flatten()))

    return num_images_list, dist_params_list, uncertainties_list

### RUN ###
num_params_list = [2,3]
all_dist_labels = ["k1", "k2", "p1", "p2", "k3"]

all_dist_params_lists = []
all_uncertainties_lists = []
num_images_list = []

for num_params in num_params_list:
    print(f"Testing with num_params={num_params}")
    
   
    num_images_list, dist_params_list, uncertainties_list = collect_dist_params_iterative(all_images, objpoints, imgpoints, num_params)
    all_dist_params_lists.append(dist_params_list)
    all_uncertainties_lists.append(uncertainties_list)

plot_distortion_parameters_with_std_dev(num_images_list, all_dist_params_lists, all_dist_labels, num_params_list, all_uncertainties_lists)
