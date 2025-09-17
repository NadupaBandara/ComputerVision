import numpy as np
import cv2
from numba import njit
from scipy import stats
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy

def compute_statistics(image):
    stats_dict = {}
    # Splits the BGR image into 3 separate 2D arrays namely Blue, Green, Red channels.
    channels = cv2.split(image)
    colors = ['R', 'G', 'B'] #a label list for assigning the stats dictionary keys
    for i, channel in enumerate(channels): #Loops through each channel (B, G, R) one by one
        stats_dict[colors[i]] = {
            'mean': np.mean(channel), #Average pixel intensity
            'mode': stats.mode(channel, axis=None).mode.item(), #Most frequent pixel intensity
            'std': np.std(channel),#Standard deviation of pixel intensity
            'max': np.max(channel),#Maximum pixel intensity
            'min': np.min(channel)#Minimum pixel intensity
        }
    return stats_dict

def compute_entropy(image):
    #Useful for understanding how visually complex the frame is
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(gray)

def linear_transform(image, alpha=1.2, beta=30): # brightness and contrast adjustment on an image
    #alplha -> contrast, beta -> brightness
    #we're apply the formula: pixel_value = alpha * pixel_value + beta to every pixel.
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def equalize_histogram(image):#
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) #YUV separates brightness (Y) from color (U and V).
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0]) #This spreads out pixel intensities and enhancing contrast images.
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)



def apply_filter(image, filter_type='blur'):
    if filter_type == 'blur':
        return cv2.GaussianBlur(image, (7, 7), 0)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'edge': #Finds edges based on intensity gradients
        return cv2.Canny(image, 100, 200)
    elif filter_type == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) #Use 64-bit float to prevent overflow from negative values
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)#ksize big means smoother edge and small means sharper.
        return cv2.magnitude(sobelx, sobely).astype(np.uint8)#Converts image values to 8-bit integers between 0 and 255
    else:
        return image  # no filter original image
    

def plot_histogram(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color): #loop runs three times, once for each color channel.
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('RGB Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

@njit
def histogram_figure_numba(np_img):#color histogram 
    # 3 list of red, green and blue values from the image
    r = np_img[:, :, 2].flatten()#count how often each pixel value appears not the location 
    g = np_img[:, :, 1].flatten()
    b = np_img[:, :, 0].flatten()
    #Three empty counters, one for each color.
    r_hist = np.zeros(256, dtype=np.int32)
    g_hist = np.zeros(256, dtype=np.int32)
    b_hist = np.zeros(256, dtype=np.int32)

    for i in range(r.shape[0]):
        r_hist[r[i]] += 1
        g_hist[g[i]] += 1
        b_hist[b[i]] += 1

    return r_hist, g_hist, b_hist #Return how often each pixel intensity appears.