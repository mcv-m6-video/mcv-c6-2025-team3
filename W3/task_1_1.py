import sys
from PIL import Image
sys.path.append('/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W3/pyflow')
import os
import time
import numpy as np
from pyflow import coarse2fine_flow
import cv2
import numpy as np

def compute_msen_pepn(flow, gt, tau=3):
    # Calculate the squared error between estimated and ground truth flow
    square_error_matrix = (flow[:, :, 0:1] - gt[:, :, 0:1]) ** 2 + (flow[:, :, 1:2] - gt[:, :, 1:2]) ** 2
    
    # Apply the validity mask from ground truth to ignore invalid flow pixels
    square_error_matrix_valid = square_error_matrix * np.stack((gt[:, :, 2], gt[:, :, 2]), axis=2)

    # Count non-occluded (valid) pixels
    non_occluded_pixels = np.sum(gt[:, :, 2] != 0)

    # Compute pixel-wise error (Euclidean error)
    pixel_error_matrix = np.sqrt(np.sum(square_error_matrix_valid, axis=2))

    # Compute MSEN (Mean Square Error Norm)
    msen = np.sum(pixel_error_matrix) / non_occluded_pixels

    # Compute PEPN (Percentage of Erroneous Pixels)
    erroneous_pixels = np.sum(pixel_error_matrix > tau)
    pepn = erroneous_pixels / non_occluded_pixels

    return msen, pepn

def read_flow_gt(flow_file):
    flow_raw = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED).astype(np.double)

    u = (flow_raw[:, :, 2] - 2**15) / 64.0
    v = (flow_raw[:, :, 1] - 2**15) / 64.0
    valid = flow_raw[:, :, 0] == 1

    # Set invalid to 0
    u[valid == 0] = 0
    v[valid == 0] = 0

    return np.stack((u, v, valid), axis=2)

def flow_to_color(flow_x, flow_y):
    h, w = flow_x.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    
    hsv[..., 0] = (angle * 180 / np.pi) / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: max
    hsv[..., 2] = np.clip(magnitude * 15, 0, 255)  # Value: speed (scaled)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def optical_flow_pyflow(im1,im2,alpha,ratio,min_width,n_outer_FP_iterations,n_inner_FP_iterations,n_SOR_iterations,col_type):
    start = time.time()
    u, v, _ = coarse2fine_flow(
        im1,
        im2,
        alpha,
        ratio,
        min_width,
        n_outer_FP_iterations,
        n_inner_FP_iterations,
        n_SOR_iterations,
        col_type,
    )
    print("Optical flow computation using pyFlow complete!")
    #To visualize the flow, we can use the function flow_to_color
    flow_img = flow_to_color(u, v)

    flow_pyflow = np.dstack((u, v))
    end = time.time()
    run_time = end - start
    return flow_pyflow, flow_img, run_time

def optical_flow_farneback(im1,im2,pyr_scale=0.5,levels=3,winsize=15,iterations=3,poly_n=5,poly_sigma=1.2,flags=0):
    start = time.time()
    flow_farneback = cv2.calcOpticalFlowFarneback(
        im1,
        im2,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    print("Optical flow computation using farneback complete!")
    end = time.time()
    run_time = end - start
    return flow_farneback, run_time

def optical_flow_off_the_shelf(output_folder):
    # Path to KITTI image sequence
    kitti_path = "/Users/andrea.sanchez/Desktop/data_stereo_flow/training/image_0/"
    images = os.listdir(kitti_path)
    sequence_45_images = [image for image in images if '000045' in image]
    flow_gt = read_flow_gt("/Users/andrea.sanchez/Desktop/data_stereo_flow/training/flow_noc/000045_10.png")

    # Create and save gif from these image sequence
    image1 = Image.open(kitti_path + sequence_45_images[0])
    image2 = Image.open(kitti_path + sequence_45_images[1])
    image1.save(output_folder / "sequence_45.gif", save_all=True, append_images=[image2], loop=0, duration=1000)

    # Load images
    img1 = cv2.imread(kitti_path + sequence_45_images[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(kitti_path + sequence_45_images[1], cv2.IMREAD_GRAYSCALE)

    img1 = np.atleast_3d(img1.astype(float) / 255.0)
    img2 = np.atleast_3d(img2.astype(float) / 255.0)

    # PyFlow Parameters
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 1
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    # Compute optical flow using pyFlow
    flow_pyflow, flow_img, pyflow_time = optical_flow_pyflow(img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
    output_path = os.path.join(output_folder, "optical_flow_pyflow.png")
    cv2.imwrite(output_path, flow_img)

    # metrics for pyflow
    msen, pepn = compute_msen_pepn(flow_pyflow, flow_gt)
    print("MSEN pyflow: ", msen)
    print("PEPN pyflow: ", pepn)
    print("Run time pyflow: ", pyflow_time)

    # Farneback Parameters
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    flags = 0

    # Compute optical flow using Farneback
    farneback_flow, farneback_time = optical_flow_farneback(img1, img2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    
    # metrics for farneback
    msen2, pepn2 = compute_msen_pepn(farneback_flow, flow_gt)
    print("MSEN farneback: ", msen2)
    print("PEPN farneback: ", pepn2)
    print("Run time farneback: ", farneback_time)







    



    