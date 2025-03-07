import sys
sys.path.append('/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W3/pyflow')
import os
import time
import numpy as np
import pyflow
from pyflow import coarse2fine_flow
import cv2

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
    flow_pyflow = np.dstack((u, v))
    print("Optical flow computation using pyFlow complete!")
    end = time.time()
    run_time = end - start
    return flow_pyflow, run_time

def optical_flow_off_the_shelf():
    # Path to KITTI image sequence
    kitti_path = "/Users/andrea.sanchez/Desktop/data_stereo_flow/training/image_0/"
    images = os.listdir(kitti_path)
    sequence_45_images = [image for image in images if '000045' in image]

    img1 = cv2.imread(kitti_path + sequence_45_images[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(kitti_path + sequence_45_images[1], cv2.IMREAD_GRAYSCALE)

    img1 = np.atleast_3d(img1.astype(float) / 255.0)
    img2 = np.atleast_3d(img2.astype(float) / 255.0)

    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # PyFlow Parameters
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 1
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    # Compute optical flow using pyFlow
    flow_pyflow, run_time = optical_flow_pyflow(img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)

    print("Optical flow pyflow: ", flow_pyflow)
    print("Run time: ", run_time)




    



    