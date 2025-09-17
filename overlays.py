# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:18:55 2021

@author: droes
"""

import numpy as np
import keyboard
import cv2 # conda install opencv
from matplotlib import pyplot as plt # conda install matplotlib


# For students
def initialize_hist_figure():
    '''
    Usually called only once to initialize the hist figure.
    Do not change the essentials of this function to keep the performance advantages.
    https://www.youtube.com/watch?v=_NNYI8VbFyY
    '''
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim([-0.5, 255.5])
    # fixed size (you can normalize your values between 0, 3 or other ranges to never exceed this limit)
    ax.set_ylim([0,3])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    def_x_line = np.arange(0, 256, 1)
    # def_y_line = np.zeros(shape=(256,))
    r_plot = ax.plot(def_x_line, def_x_line, 'r', animated=True)[0]
    g_plot = ax.plot(def_x_line, def_x_line, 'g', animated=True)[0]
    b_plot = ax.plot(def_x_line, def_x_line, 'b', animated=True)[0]
    
    return fig, ax, background, r_plot, g_plot, b_plot



def update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars):
    '''
    Uses the initialized figure to update it accordingly to the new values.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    fig.canvas.restore_region(background)        
    r_plot.set_ydata(r_bars)        
    g_plot.set_ydata(g_bars)        
    b_plot.set_ydata(b_bars)

    ax.draw_artist(r_plot)
    ax.draw_artist(g_plot)
    ax.draw_artist(b_plot)
    fig.canvas.blit(ax.bbox)
    
    

def plot_overlay_to_image(np_img, plt_figure):
    '''
    Use this function to create an image overlay.
    You must use a matplotlib figure object.
    Please consider to keep the figure object always outside code loops (performance hint).
    Use this function for example to plot the histogram on top of your image.
    White pixels are ignored (transparency effect)-
    Do not change the essentials of this function to keep the performance advantages.
    '''
    
    rgba_buf = plt_figure.canvas.buffer_rgba()
    (w, h) = plt_figure.canvas.get_width_height()
    imga = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h,w,4)[:,:,:3]
    
    # ignore white pixels
    plt_indices = np.argwhere(imga < 255)

    # add only non-white values
    height_indices = plt_indices[:,0]
    width_indices = plt_indices[:,1]
    
    np_img[height_indices, width_indices] = imga[height_indices, width_indices]

    return np_img



def plot_strings_to_image(np_img, list_of_string, text_color=(255,0,0), right_space=400, top_space=50):
    '''
    Plots the string parameters below each other, starting from top right.
    Use this function for example to plot the default image characteristics.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    y_start = top_space
    min_size = right_space
    line_height = 20
    (h, w, c) = np_img.shape
    if w < min_size:
        raise Exception('Image too small in width to print additional text.')
        
    if h < top_space + line_height:
        raise Exception('Image too small in height to print additional text.')
    
    y_pos = y_start
    x_pos = w - min_size

    for text in list_of_string:
        if y_pos >= h:
            break
        # SLOW!
        np_img = cv2.putText(cv2.UMat(np_img), text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        y_pos += line_height

    if type(np_img) is cv2.UMat:
        np_img = np_img.get()

    return np_img


#Special Task: Meme Face

#Loads OpenCV’s built-in Haar Cascade classifier for detecting frontal faces.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

meme_faces = [] # List to store the loaded meme images.
current_meme_index = [0]  # Mutable list for access in loop

def load_memes(paths):
    """Load meme PNGs with alpha channel."""
    global meme_faces
    meme_faces = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in paths]

def overlay_meme_on_face(frame):
    """Overlay the current meme face on all detected faces."""
    if not meme_faces:
        return frame

    # Switch meme with 'm' key
    if keyboard.is_pressed('m'):
        current_meme_index[0] = (current_meme_index[0] + 1) % len(meme_faces)
        print(f"[Meme] Switched to: meme{current_meme_index[0]+1}.png")
        cv2.waitKey(300)  # debounce delay

    meme = meme_faces[current_meme_index[0]]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Haar cascades work on intensity values, grayscale makes faster
    
    '''
    Detecting the face:

    detectMultiScale()-> Detects faces in the image.
    parameters:
    (gray)- grayscale image
    (1.3)-How much the image size is reduced at each scale. smaller is more accurate but slower. 30%
    (5)- How many neighbors each rectangle should have to consider it a face. 

    '''
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #Each detected face is represented as a rectangle:
    for (x, y, w, h) in faces:
        new_w, new_h = int(w * 1.5), int(h * 1.5) #Enlarging the memes to 1.5x to better overlaying
        meme_resized = cv2.resize(meme, (new_w, new_h), interpolation=cv2.INTER_AREA) #Meme enlargement
        x_offset = x - int((new_w - w) / 2)#Shift the top-left x position slightly left to center the enlarged meme on the detected face.
        y_offset = y - int((new_h - h) / 2)#same for vertical centering.
        frame = blend_overlay(frame, meme_resized, x_offset, y_offset)

    return frame

def blend_overlay(background, overlay_rgba, x, y):
    """This function blends a transparent image (RGBA) onto a background image (BGR) at position (x, y) """
    h, w = overlay_rgba.shape[:2] #Extracts the height and width of the overlay image.
    #Bound check to prevent the overlay from going outside of visible area.
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    overlay_rgb = overlay_rgba[:, :, :3] #takes the R, G, B channels without the transparency alpha.
    alpha = overlay_rgba[:, :, 3] / 255.0 #takes the alpha channel and normalizes between 0-1.
    region = background[y:y+h, x:x+w] #Selects the exact region in the background where the meme will be pasted.
    '''Formula for pixel:
    
    blended_pixel = (1 - transparency) * background_pixel + transparency * overlay_pixel
    [:, :, None] -> Add a new dimension to the array to match the number of channels in the background image.
    '''
    blended = (1 - alpha[:, :, None]) * region + alpha[:, :, None] * overlay_rgb
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background