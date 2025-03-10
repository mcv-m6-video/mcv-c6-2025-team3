from PIL import Image
import time
import numpy as np
from pyflow import coarse2fine_flow
import cv2
import matplotlib.pyplot as plt
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter



def compute_msen_pepn(flow, gt, valid_gt, tau=3):
    # Get x and y values
    x_gt = gt[:,:,0]
    y_gt = gt[:,:,1]
    x_flow = flow[:,:,0]
    y_flow = flow[:,:,1]

    # is calculated for every pixel
    motion_vectors = np.sqrt( np.square(x_flow - x_gt) + np.square(y_flow - y_gt) )

    # erroneous pixels are the ones where motion_vector > 3 and are valid pixels 
    err_pixels = np.sum(motion_vectors[valid_gt] > tau)

    # calculate metrics
    msen = np.mean(motion_vectors[valid_gt])
    pepn = (err_pixels / np.sum(valid_gt)) * 100 # erroneous pixels / total valid pixels from the ground truth

    return msen, pepn, motion_vectors

def read_flow_gt(flow_file):
    flow_raw = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED).astype(np.float32)

    u = (flow_raw[:, :, 2] - 2**15) / 64.0
    v = (flow_raw[:, :, 1] - 2**15) / 64.0
    valid = flow_raw[:, :, 0] == 1

    # Set invalid to 0
    u[valid == 0] = 0
    v[valid == 0] = 0

    return np.stack((u, v), axis=2), valid

def flow_to_color(flow_x, flow_y):
    h, w = flow_x.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: max
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: speed (scaled), normalized

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def generate_optical_flow_legend(output_folder):    
    size = 300  
    flow_x, flow_y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size)) 

    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180 / np.pi) / 2  
    hsv[..., 1] = 255 
    hsv[..., 2] = 255  

    legend_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(legend_image)
    ax.axis('off')

    arrow_x = size // 2
    arrow_y = size // 2
    arrow_length = size // 4  # Tamaño de la flecha
    ax.arrow(arrow_x, arrow_y, arrow_length, 0, head_width=15, head_length=15, fc='black', ec='black', lw=2)
    ax.arrow(arrow_x, arrow_y, -arrow_length, 0, head_width=15, head_length=15, fc='black', ec='black', lw=2)
    ax.arrow(arrow_x, arrow_y, 0, -arrow_length, head_width=15, head_length=15, fc='black', ec='black', lw=2)
    ax.arrow(arrow_x, arrow_y, 0, arrow_length, head_width=15, head_length=15, fc='black', ec='black', lw=2)

    ax.text(size//2, 10, "Top", fontsize=12, color='white', ha='center', va='top')
    ax.text(size//2, size-10, "Bottom", fontsize=12, color='white', ha='center', va='bottom')
    ax.text(10, size//2, "Left", fontsize=12, color='white', ha='left', va='center', rotation=90)
    ax.text(size-10, size//2, "Right", fontsize=12, color='white', ha='right', va='center', rotation=-90)

    plt.savefig(f'{output_folder}/optical_flow_legend.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def optical_flow_pyflow(im1,im2,alpha,ratio,min_width,n_outer_FP_iterations,n_inner_FP_iterations,n_SOR_iterations,col_type):
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

    return flow_pyflow, flow_img

def optical_flow_farneback(im1,im2,pyr_scale=0.5,levels=3,winsize=15,iterations=3,poly_n=5,poly_sigma=1.2,flags=0):
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
    flow_img = flow_to_color(flow_farneback[..., 0], flow_farneback[..., 1])
    return flow_farneback, flow_img


def optical_flow_off_the_shelf(output_folder, img1_path, img2_path, option, subset):
    # load image in rgb and create and save gif from these image sequence
    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)
    image1.save(f'{output_folder}/sequence_45_{subset}.gif', save_all=True, append_images=[image2], loop=0, duration=1000)

    if option == 'pyflow':
        #normalize for pyflow, 3 dimensions expected, double
        img1 = np.atleast_3d(np.array(image1).astype(np.float64) / 255.0)
        img2 = np.atleast_3d(np.array(image2).astype(np.float64) / 255.0)

        # PyFlow Parameters
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 1
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        # Compute optical flow using pyFlow
        start = time.time()
        flow, flow_color = optical_flow_pyflow(img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
        end = time.time()
        run_time = end - start

    elif option == 'farneback':
        #normalice for farneback, 2 dimensions expected, uint 8
        img1 = np.array(image1)
        img2 = np.array(image2)

        if len(img1.shape) == 3:  
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        # Farneback Parameters
        pyr_scale = 0.5
        levels = 3
        winsize = 15
        iterations = 3
        poly_n = 5
        poly_sigma = 1.2
        flags = 0

        # Compute optical flow using Farneback
        start = time.time()
        flow, flow_color = optical_flow_farneback(img1, img2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        end = time.time()
        run_time = end - start

    elif option in ['raft', 'craft', 'memflow', 'maskflownet']: #https://ptlflow.readthedocs.io/en/latest/models/models_list.html
        img1 = np.array(image1)
        img2 = np.array(image2)

        #ensure rgb
        if len(img1.shape) == 2:  
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


        model = ptlflow.get_model(f"{option}", ckpt_path='kitti')

        io_adapter = IOAdapter(model, img1.shape[:2])
        data = io_adapter.prepare_inputs([img1, img2])

        start = time.time()
        output = model(data)
        end = time.time()
        run_time = end - start

        flow = output['flows']
        flow = np.squeeze(flow)  
        flow = flow.permute(1, 2, 0)
        flow = flow.detach().cpu().numpy()

        flow_color = flow_to_color(flow[..., 0], flow[..., 1])


    cv2.imwrite(f'{output_folder}/optical_flow_{option}_{subset}.png', flow_color)

    return flow, run_time


