import cv2
import numpy as np
from skimage import data
from skimage.util import img_as_ubyte
from skimage import exposure
import skimage.morphology as morp
from skimage.filters import rank
import matplotlib.pyplot as plt
from suace import perform_suace
def finger_print():
    image = cv2.imread('finger1.bmp',cv2.IMREAD_GRAYSCALE)
    flag = 0
    _, mask = cv2.threshold(image, thresh=50,maxval=255, type=cv2.THRESH_BINARY)
    im_thresh_gray = cv2.bitwise_and(image, mask)
    t_lower = 150  # Lower Threshold 
    t_upper = 250  # Upper threshold 
    point = []
    thresh1 = cv2.GaussianBlur(im_thresh_gray, (7, 7), 0)

    edge = cv2.Canny(thresh1, t_lower, t_upper)
    for i in range(0,240):
        if edge[i,160] == 255 and flag == 0:
            edge[i,160] = 255
            point.append([i,160])
            flag = 1
        elif edge[i,160] == 0 and flag == 1:
            edge[i,160] = 255
            flag = 1
        elif edge[i,160] == 255 and flag == 1:
            edge[i,160] = 255
            point.append([i,160])
            flag = 0
        else:
            flag = 0
    image_cropped = image[point[0][0] : point[1][0] ,0:320]
    result = perform_suace(image_cropped, distance=1, sigma=9.0)

    img_global = 0
    clahe = cv2.createCLAHE(clipLimit=5)
    image_cropped = clahe.apply(image_cropped) + 25
    img_global = exposure.equalize_hist(image_cropped)

    kernel = morp.disk(30)
    img_local = rank.equalize(img_global, selem=kernel)

    # print(img_local.shape)
    cropped_image = cv2.resize(result, (313, 126)) 
    cropped_image = cv2.medianBlur(cropped_image,5)
    thresh1 = cv2.adaptiveThreshold(cropped_image, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 199, 5)
    thresh1 = cv2.resize(thresh1, (313, 126)) 
    return thresh1,img_local,image_cropped,result,

def knuckle_print():
    
    image = cv2.imread('knuckle7.bmp',cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=5)
    image_cropped = clahe.apply(image) + 25
    img_global = exposure.equalize_hist(image_cropped)

    kernel = morp.disk(30)
    img_local = rank.equalize(img_global, selem=kernel)
    thresh1 = cv2.adaptiveThreshold(img_local, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 199, 5)
    thresh1 = cv2.resize(thresh1, (313, 126)) 

    return thresh1
thresh1,image,image_cropped,img_global = finger_print()
# thresh2 = knuckle_print()

# fused_image = np.logical_xor(thresh1, thresh2)
# fused_image = fused_image.astype(np.uint8) * 255
figure, axis = plt.subplots(2, 2) 

axis[0, 0].imshow(image_cropped,cmap = 'gray') 
axis[0,0].set_title('Original Image')


axis[1,0].imshow(img_global,cmap ='gray') 
axis[1,0].set_title('suace Image')

axis[1,1].imshow(image,cmap ='gray') 
axis[1,1].set_title('CLAHE Image')








plt.show()





















