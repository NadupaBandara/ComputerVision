# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""

import numpy as np
import keyboard
import cv2
from capturing import VirtualCamera
from overlays import (
    initialize_hist_figure,
    plot_overlay_to_image,
    plot_strings_to_image,
    update_histogram,
    load_memes, 
    overlay_meme_on_face 
)
from basics import (
    histogram_figure_numba,
    linear_transform,
    apply_filter,
    compute_statistics,
    compute_entropy
)
# Global mutable filter setting
current_filter = ['none']  # default filter

#This function processes each frame from the image source
def custom_processing(img_source_generator):
    # Sets up a live histogram to display RGB pixel distributions.
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    #Grabs one frame at a time from webcam 
    for sequence in img_source_generator:
        # --- Handle keypresses ---
        if keyboard.is_pressed('1'):
            current_filter[0] = 'blur'
        elif keyboard.is_pressed('2'):
            current_filter[0] = 'sharpen'
        elif keyboard.is_pressed('3'):
            current_filter[0] = 'edge'
        elif keyboard.is_pressed('4'):
            current_filter[0] = 'sobel'
        elif keyboard.is_pressed('5'):
            current_filter[0] = 'equalize'
        elif keyboard.is_pressed('0'):
            current_filter[0] = 'none'

        # --- Linear transform ---
        sequence = linear_transform(sequence, alpha=1.2, beta=20)

        # --- Apply filter if selected ---
        if current_filter[0] != 'none':
            sequence = apply_filter(sequence, filter_type=current_filter[0])
            # Ensure the image is BGR (OpenCV uses BGR) if the image is grayscale.
            if len(sequence.shape) == 2:
                sequence = cv2.cvtColor(sequence, cv2.COLOR_GRAY2BGR)

        #Only show meme if no filter is applied
        if current_filter[0] == 'none':
            sequence = overlay_meme_on_face(sequence)

        # --- Histogram overlay ---
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        sequence = plot_overlay_to_image(sequence, fig)

        # --- Text overlay ---
        stats = compute_statistics(sequence)
        entropy_val = compute_entropy(sequence)
        display_text_arr = [
            "Computer Vision Project",
            f"Filter: {current_filter[0].capitalize() if current_filter[0] != 'none' else 'Off'}",
            f"Entropy: {entropy_val:.2f}",
            f"R: mean={stats['R']['mean']:.1f} std={stats['R']['std']:.1f}",
            f"G: mean={stats['G']['mean']:.1f} std={stats['G']['std']:.1f}",
            f"B: mean={stats['B']['mean']:.1f} std={stats['B']['std']:.1f}",
        ]
        sequence = plot_strings_to_image(sequence, display_text_arr)
        #This ensures pixel values are safe (valid range: 0–255).
        sequence = np.clip(sequence, 0, 255).astype(np.uint8)

        yield sequence

def main():
    #Initialize camera parameters for webcam  
    width = 1280
    height = 720
    fps = 30

    vc = VirtualCamera(fps, width, height)
    load_memes([
    "meme_faces/doge.png",
    "meme_faces/sad.png",
    "meme_faces/smug.png"
    ])
    vc.virtual_cam_interaction(
        custom_processing(
            vc.capture_cv_video(0, bgr_to_rgb=False)
        )
    )


if __name__ == "__main__":
    main()