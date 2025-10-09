# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:41 2021

@author: droes
"""
import basics
#import overlays  # later, if used
import pyvirtualcam
import numpy as np
import cv2 # conda install opencv
from PIL import ImageGrab # conda install pillow
from matplotlib import pyplot as plt # conda install matplotlib
import keyboard


class VirtualCamera:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width = width
        self.height = height
        
    def capture_screen(self, plt_inside=False, alt_width=0, alt_height=0):
        '''
        Represents the content of the primary monitor.
        Can be used to quickly test your application.
        '''
        
        width = alt_width if alt_width > 0 else self.width
        height = alt_height if alt_height > 0 else self.height
        while True:
            # grab is a slow method!
            img = ImageGrab.grab(bbox=(0, 0, width, height)) #x, y, w, h
            img_np = np.array(img)
            #img_np = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            if plt_inside:
                plt.imshow(img_np)
                plt.axis('off')
                plt.show()
            yield img_np

            
    def capture_cv_video(self, camera_id, bgr_to_rgb=False):
       
        cv_vid = cv2.VideoCapture(camera_id)

        if not cv_vid.isOpened():
            raise RuntimeError('Video-Output cannot be opened.')
            
        cv_vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cv_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv_vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cv_vid.set(cv2.CAP_PROP_FPS, self.fps)

        # Tatsächliche Einstellungen können sich von den oberhalb festgelegten dennoch unterscheiden!
        width = int(cv_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cv_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cv_vid.get(cv2.CAP_PROP_FPS)
        print(f'Camera properties: ({width}x{height} @ {fps_in}fps)')
        
        while True:
            ret, frame = cv_vid.read()
            if not ret:
                raise RuntimeError('Camera image cannot be loaded.')
            if bgr_to_rgb:
                frame = frame[...,::-1]
                
            if keyboard.is_pressed('q'):
                # quit camera stream
                cv_vid.release()
                return
                
            yield frame

    
    def virtual_cam_interaction(self, img_generator, print_fps=True):
        '''
        Provides a virtual camera.
        img_generator must represent a function that acts as a generator and returns image data.
        '''
        print('Quit camera stream with "q"')
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, print_fps=print_fps) as cam:
           # Capture frames from the generator and send them to the virtual camera.
            for img in img_generator:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cam.send(rgb_img)
                cam.sleep_until_next_frame()

