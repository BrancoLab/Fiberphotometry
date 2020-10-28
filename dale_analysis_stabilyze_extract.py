from fcutils.video import utils
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d, derivative
from fcutils.plotting.utils import clean_axes, save_figure
import imutils
import matplotlib.pyplot as plt
import pandas as pd
from fcutils.file_io.utils import listdir, get_file_name
import os 
from scipy.signal import medfilt

from behaviour.tracking.tracking import prepare_tracking_data
from fcutils.maths.filtering import smooth_hanning
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as calc_angles

"""
    Script to analyse preliminary data set by Dale as it fixes the setup. 

    What it does:
     - threshold + contours to get the center of the fiber bundle at each frame
     - uses that to crop frame around bundle, removing translation
     - threshold + contour again to find the brightest fiber at each frame
            this is used to compute and remove frame rotation
     - signal extraction by hand defined ROIs and frame masking
     - plotting
"""

# --------------------------------- settings --------------------------------- #
RENAME_FRAMES = False

LOAD_FROM_IMAGES = False
SAVE_THRESH_VIDEO = False
CROP = False
STABILIZE = True
EXTRACT = False

ANALYSE = False

# ----------------------------------- paths ---------------------------------- #
N_FIBERS = 8

fld = Path(r'D:\Dropbox (UCL)\Photometry\20201015\405nm rotations')
raw_img_name = 'frame_'

basename = fld.parent.name
analysis_fld = fld.parent / f'analysis {fld.name}'
analysis_fld.mkdir(exist_ok=True)

main_video = str( analysis_fld/ f'{basename}.mp4')
threshvideo = str(analysis_fld / f'{basename}_th.mp4')
thresh_cropped_video = str(analysis_fld / f'{basename}_th_crop.mp4')
cropped_video = str(analysis_fld / f'{basename}_crop.mp4')
stable_video = str(analysis_fld / f'{basename}_stable.mp4')
rois_video = str(analysis_fld / f'{basename}_stable_rois.mp4')

anglespath = str(analysis_fld / f'{basename}_angles.npy')
datapath = str(analysis_fld / f'{basename}_data.h5')
figpath = str(analysis_fld / f'{basename}_signal')

# --------------------------------- variables -------------------------------- #
fps= 25
TH = 40 # 70  # 40 - used for detecting fiber bundle for cropping
TH2 = 150 # 120 - used to detect contours for rotation
crop = 350 if N_FIBERS == 4 else 450  #450 # 350
radius = 80

ANGLE_SHIFT = 5 # to manually offset the rotation-corrected frames 

if N_FIBERS == 4:
    centers = [
        (350, 140),
        (525, 325),
        (350, 500),
        (165, 325)
    ]

    colors = {
        'roi-0':(255, 0, 0),
        'roi-1':(0, 255, 0),
        'roi-2':(0, 0, 255),
        'roi-3':(128, 128, 0)
    }

elif N_FIBERS == 8:
    centers = [
        (330, 180),
        (750, 390),
        (470, 535),
        (170, 460),
        (245, 720),
        (480, 790),
        (690, 650),
        (580, 200),
    ]


    colors = {
        'roi-0':(255, 0, 0),
        'roi-1':(0, 255, 0),
        'roi-2':(0, 0, 255),
        'roi-3':(128, 128, 0),
        'roi-4':(0, 128, 255),
        'roi-5':(128, 0, 128),
        'roi-6':(200, 200, 200),
        'roi-7':(200, 100, 60)
    }

# ---------------------------------- Rename ---------------------------------- #
if RENAME_FRAMES:
    files_list = [f for f in listdir(str(fld)) if '.tiff' in f]
    files = sorted(files_list, key = lambda x: (int(x.split('_')[-1].split('.')[0])))
    for n, f in enumerate(files):
        # name = get_file_name(f)
        # new_name = 'frame_'+ name.split('_')[-1] + '.tiff'
        new_name = f'frame_{n}.tiff'
        os.rename(f, os.path.join(str(fld), new_name))

# ----------------------------------- load ----------------------------------- #

# # Load video from folder
if LOAD_FROM_IMAGES:
    print('Making video from frames')
    cap = utils.get_cap_from_images_folder(str(fld), 
                img_format = f'{raw_img_name}%1d.tiff')

    # Save as video
    utils.save_videocap_to_video(cap,main_video, 'mp4', fps=fps)

# --------------------------------- threshold -------------------------------- #

if SAVE_THRESH_VIDEO:
    # Iterate over frames
    cap = utils.get_cap_from_file(main_video)
    nframes, width, height, fps, is_color = utils.get_video_params(cap)

    writer =  utils.open_cvwriter( threshvideo, 
            w=width, h=height, framerate=fps, format=".mp4", iscolor=False)


    for framen in tqdm(np.arange(nframes)):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold
        frame = cv2.GaussianBlur(frame,(15,15),0)
        ret, frame = cv2.threshold(frame, TH, 255,cv2.THRESH_BINARY)

        # # # morphological transformations
        kernel = np.ones((15,15),np.uint8)
        frame = cv2.erode(frame, kernel, iterations = 2)
        frame = cv2.dilate(frame, kernel, iterations = 4)

        # Threshold again
        frame = cv2.GaussianBlur(frame,(91,91),0)
        frame = cv2.GaussianBlur(frame,(91,91),0)
        frame = cv2.GaussianBlur(frame,(91,91),0)

        ret, frame = cv2.threshold(frame, TH, 255,cv2.THRESH_BINARY)

        show = frame.copy()
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)



        hull=cv2.convexHull(c)
        cent, axes, rad = cv2.fitEllipse(hull)
        
        cv2.ellipse(show,cv2.fitEllipse(hull),(0,255,0),2)
        cv2.rectangle(show,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(show, tuple([int(x) for x in cent]), 10, (0, 0, 255), -1)

        cv2.imshow('thresholding', show)
        cv2.waitKey(1)
        writer.write(frame)
    cap.release()
    writer.release()

# --------------------------------- crop -------------------------------- #
if CROP:
    cap = utils.get_cap_from_file(threshvideo)
    raw_cap = utils.get_cap_from_file(main_video)

    nframes, width, height, fps, is_color = utils.get_video_params(cap)

    
    writer =  utils.open_cvwriter( cropped_video, 
            w=crop * 2, h=crop * 2, framerate=fps, format=".mp4", iscolor=True)
    writer2 =  utils.open_cvwriter( thresh_cropped_video, 
            w=crop * 2, h=crop * 2, framerate=fps, format=".mp4", iscolor=True)

    for framen in tqdm(np.arange(nframes)):
        ret, frame = cap.read()
        original = frame.copy()
        ret, rawframe = raw_cap.read()
        if not ret: break

        # Get contour of fiber bundle
        # if framen == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)

        # Get center of fiber bundle
        # Get biggest contour
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Get center
        M = cv2.moments(c)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        cv2.circle(frame, (x, y), 10, (0, 0, 255), 3) 

        # Crop around bundle
        cropped = rawframe[y-crop:y+crop, x-crop:x+crop]
        cropped2 = original[y-crop:y+crop, x-crop:x+crop]

        cv2.imshow('frame', cropped)
        cv2.waitKey(2)
        writer.write(cropped)
        writer2.write(cropped2)
    cap.release()
    writer.release()
    writer2.release()


# --------------------------------- stablize --------------------------------- #
if STABILIZE:
    # Check if DLC file exists
    print(analysis_fld)
    dlc_output = [f for f in analysis_fld.glob('*.h5')]
    if not dlc_output:
        raise FileNotFoundError('Looks like you need to run DLC on the video, Fede')

    # Get angles 
    tracking = prepare_tracking_data(str(dlc_output[0]), likelihood_th=.1)
    angles = np.degrees(np.unwrap(np.radians(calc_angles(
        tracking['center-center'].x,
        tracking['center-center'].y,
        tracking['center-out'].x,
        tracking['center-out'].y,
    ))))

    
    angles = smooth_hanning(angles)
    x = smooth_hanning(tracking['center-center'].x.values)
    y = smooth_hanning(tracking['center-center'].y.values)
    plt.plot(x)
    plt.plot(y)
    plt.plot(angles)
    plt.show()

    # Setup opencv writer
    cap = utils.get_cap_from_file(cropped_video)
    nframes, width, height, fps, is_color = utils.get_video_params(cap)
    writer =  utils.open_cvwriter(stable_video, 
            w=crop * 2, h=crop * 2, framerate=fps, format=".mp4", iscolor=True)

    # Save video
    for angle, xx, yy in zip(angles, x, y):
        ret, frame = cap.read()
        if not ret: break
        original = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # translate
        center = np.int(frame.shape[0]/2)
        dx = - (xx - center)
        dy = - (yy - center)

        original = imutils.translate(original, dx, dy)

        # Rotate frame
        rotated = imutils.rotate(original, - angle)


        cv2.imshow('original', rotated)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)
        writer.write(rotated)

    cap.release()
    writer.release()

    np.save(anglespath, angles)


# ---------------------------------- extract --------------------------------- #
if EXTRACT:
    dlc_output = [f for f in analysis_fld.glob('*.h5')]
    if not dlc_output:
        raise FileNotFoundError('Looks like you need to run DLC on the video, Fede')
    tracking = prepare_tracking_data(str(dlc_output[0]), likelihood_th=.1)
    
    cap = utils.get_cap_from_file(stable_video)
    nframes, width, height, fps, is_color = utils.get_video_params(cap)
    writer =  utils.open_cvwriter(rois_video, 
            w=crop * 2, h=crop * 2, framerate=fps, format=".mp4", iscolor=True)

    signals = {'roi-'+str(i):[] for i in np.arange(N_FIBERS)}
    for framen in tqdm(np.arange(nframes)):
        ret, frame = cap.read()
        original = frame.copy()
        if not ret: break
        if framen < 30: continue

        frame = imutils.rotate(frame, ANGLE_SHIFT)

        # Get mask
        masks = {'roi-'+str(i):np.zeros_like(frame) for i in np.arange(N_FIBERS)}
        for n, cent in enumerate(centers, ):
            col = colors['roi-'+str(n)]

            cv2.circle(frame, cent, radius, col, 3)
            cv2.circle(masks['roi-'+str(n)], cent, radius, (255, 255, 255), -1)

            frame = cv2.putText(frame, f'ROI - {n}', cent, cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, col, 2, cv2.LINE_AA)
            
        for name, mask in masks.items():
            masked = original.copy().astype(np.float32)
            # masked = (original * mask).astype(np.float32)
            masked[mask == 0] = np.nan
            signals[name].append(np.nanmean(masked))

            # if name == 'roi-5': 
            #     show = original * mask

            #     f, axarr = plt.subplots(ncols=3)
            #     axarr[0].imshow(original)
            #     axarr[0].imshow(mask, cmap='Reds', alpha=.5)
            #     axarr[1].imshow(mask)
            #     axarr[2].imshow(masked)
            #     plt.show()

            if framen == 0:
                cv2.imwrite(str(analysis_fld/ f'{name}_mask.png'), mask) 

        # if framen > 100: break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(10)
        writer.write(frame)

        if key==27:    # Esc key to stop
            break

    cap.release()
    writer.release()

    data = pd.DataFrame(signals)
    try:
        data['angles'] = np.load(anglespath)
    except:
        data['angles'] = np.zeros(len(data))
    data.to_hdf(datapath, key='hdf')


if ANALYSE:
    data = pd.read_hdf(datapath, key='hdf')
    print(data.mean())

    f, axarr = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, gridspec_kw = dict(height_ratios =[2, 1]))
    f2, axarr2 = plt.subplots(nrows=N_FIBERS, figsize=(8, 3*N_FIBERS), sharex=False)

    for col in data.columns:
        if not 'roi-' in col: continue
        color = np.array([c/255 for c in colors[col]])[::-1]

        sig = medfilt(data[col], 9)

        axarr[0].plot(sig, color='w', lw=5)    
        axarr[0].plot(sig, color=color, lw=3, label=col)    

        mn = np.mean(sig)
        bound = mn * .05

        axarr[0].axhline(mn-bound, ls=':', color=color, alpha=.7)
        axarr[0].axhline(mn+bound, ls=':', color=color, alpha=.7)

        axarr2[int(col[-1])].hist(sig, bins=5, color=color, label=col, alpha=.4)
        axarr2[int(col[-1])].hist(sig, bins=5, color=color, label=col, alpha=1, histtype='step')
        axarr2[int(col[-1])].axvline(mn-bound, ls=':', color=color, alpha=.7)
        axarr2[int(col[-1])].axvline(mn+bound, ls=':', color=color, alpha=.7)
        axarr2[int(col[-1])].axvline(mn, color='white', alpha=1, lw=4)
        axarr2[int(col[-1])].axvline(mn, color=color, alpha=1, lw=2)
        axarr2[int(col[-1])].set(ylabel=col)


    avel = derivative(np.unwrap(np.radians(data.angles)))
    avel = np.degrees(avel) * fps
    axarr[1].plot(avel)

    f.suptitle(basename)
    axarr[0].set(ylabel='ROI signal (med filt)')
    axarr[0].legend()
    axarr[1].set(ylabel='angular velocity (degrees per sec)', xlabel='frames')

    clean_axes(f)
    save_figure(f, figpath)
    
    clean_axes(f2)
    save_figure(f2, figpath+'_hist')
    f2.tight_layout()
    plt.show()
