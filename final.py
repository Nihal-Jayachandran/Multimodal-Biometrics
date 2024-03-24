import cv2
import numpy as np
from skimage import data
from skimage.util import img_as_ubyte
from skimage import exposure
import skimage.morphology as morp
from skimage.filters import rank
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from suace import perform_suace
#Load the pre-trained VGG16 model

def cnn():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Function to extract features from an image
    def extract_features(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        features = base_model.predict(img_array)
        return features.flatten()
    k = [1,2,4,5,6,8]
    similarity_score = []
    for i in k:
        for j in k:
            if i ==j or i>j:
                continue
            else:
                identifier_1 = extract_features(f"fused_image{i}.bmp")
                identifier_2 = extract_features(f"fused_image{j}.bmp")
                similarity = cosine_similarity([identifier_1], [identifier_2])[0][0]
                similarity_score.append([(i,j),similarity])
# Output similarity score
    print("Similarity between the images:", similarity_score)
#cnn()
# def finger_print(distinct_random_values):

def calculate_histogram(tile, bins=256):
    return np.histogram(tile.flatten(), bins, range=(0, 256))[0]

def clip_histogram(hist, clip_limit):
    total_excess = sum(hist[hist > clip_limit] - clip_limit)
    bin_increment = total_excess // hist.size
    upper_limit = clip_limit + bin_increment

    hist[hist > clip_limit] = clip_limit
    hist[hist < upper_limit] += int(bin_increment)

    return hist


def performSUACE(src, dst, distance=20, sigma=7, tile_grid_size=(10, 10), clip_limit=5.0):
    assert src.dtype.name == 'uint8', "Input image must be of type CV_8UC1"  # Check dtype
    dst = np.zeros_like(src)

    smoothed = cv2.GaussianBlur(src, (0, 0), sigma)

    distance_d = np.float64(distance)
    half_distance = np.uint8(distance // 2) 

    for x in range(src.shape[1]):
        for y in range(src.shape[0]):
            val = np.float64(src[y, x])
            adjuster = np.float64(smoothed[y, x])

            if (val - adjuster) > distance_d:
                adjuster += (val - adjuster) * 0.5

            adjuster = max(adjuster, half_distance)
            b = adjuster + half_distance
            b = min(b, 255)
            a = b - distance
            a = max(a, 0)

            if val >= a and val <= b:
                tile_y_start = max(0, y - tile_grid_size[1] // 2)
                tile_y_end = min(src.shape[0], y + tile_grid_size[1] // 2)
                tile_x_start = max(0, x - tile_grid_size[0] // 2)
                tile_x_end = min(src.shape[1], x + tile_grid_size[0] // 2)

                tile = src[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

                hist = calculate_histogram(tile)
                clipped_hist = clip_histogram(hist, clip_limit)

                cum_dist = clipped_hist.cumsum()
                cum_dist_scaled = (cum_dist * 255 / cum_dist.max()).astype(np.uint8)

                tile = cum_dist_scaled[tile]

                dst[y, x] = int(((val - a) / distance_d) * tile[y - tile_y_start, x - tile_x_start] + (1 - (val - a) / distance_d) * val)
            elif val < a:
                dst[y, x] = 0
            elif val > b:
                dst[y, x] = 255

    return dst

def highlight_finger_veins(image_path, distance=20, sigma=7, blur_kernel_size=11):

    distance = 21
    sigma = 36
    tile_grid_size = (10, 10)
    clip_limit = 1.0

    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    denoised_image = cv2.medianBlur(image_path, blur_kernel_size)
    enhanced_image = performSUACE(denoised_image, np.zeros_like(denoised_image), distance, sigma / 8.0)
    #_, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #result_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    return enhanced_image

# # Define the range and the number of distinct random values you want

def finger_print():
    for k in range(1,100):
        try:
            print("start" + str(k))
            image = cv2.imread(f'finger{k}.bmp',cv2.IMREAD_GRAYSCALE)
            flag = 0
            _, mask = cv2.threshold(image, thresh=50,maxval=255, type=cv2.THRESH_BINARY)
            im_thresh_gray = cv2.bitwise_and(image, mask)
            t_lower = 0  # Lower Threshold
            t_upper = 150  # Upper threshold
            point = []
            thresh1 = cv2.GaussianBlur(im_thresh_gray, (7, 7), 0)
            edge = cv2.Canny(thresh1, t_lower, t_upper)
            # print(image.shape)
            # print(edge.shape)
            # plt.imshow(edge, cmap='gray')
            # plt.show()
            for i in range(0, min(240, edge.shape[0])):
                #print(i)
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
            #clahe = cv2.createCLAHE(clipLimit=5)
            #image_cropped = clahe.apply(image_cropped) + 25
            image_cropped = highlight_finger_veins(image_cropped)
            #img_global = exposure.equalize_hist(image_cropped)

            #kernel = morp.disk(30)
            #img_local = rank.equalize(img_global, selem=kernel)
            cropped_image = cv2.resize(image_cropped, (313, 126)) 
            cropped_image = cv2.medianBlur(cropped_image,5)
            thresh1 = cv2.adaptiveThreshold(cropped_image, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 199, 5)
            thresh1 = cv2.resize(thresh1, (313, 126)) 

            image = cv2.imread(f'knuckle{k}.bmp',cv2.IMREAD_GRAYSCALE)
            #clahe = cv2.createCLAHE(clipLimit=5)
            #image_cropped = clahe.apply(image) + 25
            image_cropped = highlight_finger_veins(image)
            #img_global = exposure.equalize_hist(image_cropped)

            #kernel = morp.disk(30)
            #img_local = rank.equalize(img_global, selem=kernel)
            thresh2 = cv2.adaptiveThreshold(image_cropped, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 199, 5)
            thresh2 = cv2.resize(thresh2, (313, 126)) 

            fused_image = np.maximum(thresh1, thresh2)
            fused_image = fused_image.astype(np.uint8) * 255

            cv2.imwrite(f'fused_image{k}.bmp',fused_image)
            
        except Exception as e:
            print(f"Error processing image {k}: {e}")
            continue

start_range = 1
end_range = 101
num_values = 9  # Adjust this according to your needs

# Generate a list of distinct random values
distinct_random_values = random.sample(range(start_range, end_range + 1), num_values)

#finger_print()
cnn()