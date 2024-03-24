import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage import data
from skimage.util import img_as_ubyte
from skimage import exposure
import skimage.morphology as morp
from skimage.filters import rank

class HoneyBadger:
    def __init__(self, num_dimensions, search_space):
        self.position = np.random.uniform(search_space[0], search_space[1], num_dimensions)
        self.best_position = np.copy(self.position)
        self.fitness = float('inf')  # Initialize with infinity

def evaluate_gabor_fitness(gabor_parameters, image,similarity_score_list):
    filtered_finger_vein2 = apply_gabor_filter(image)
    filtered_finger_vein3 = apply_gabor_filter(image,gabor_parameters[0],gabor_parameters[1],gabor_parameters[2],gabor_parameters[3],gabor_parameters[4])



    thresh1 = cv2.GaussianBlur(filtered_finger_vein2, (7, 7), 0)
    thresh1 = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 199, 5)

    thresh2 = cv2.GaussianBlur(filtered_finger_vein3, (7, 7), 0)
    thresh2 = cv2.adaptiveThreshold(thresh2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 199, 5)
    fused_features_flattened_1 = thresh1.flatten()
    fused_features_flattened_2 = thresh2.flatten()
    fused_features_normalized_1 = fused_features_flattened_1 / np.linalg.norm(fused_features_flattened_1)
    fused_features_normalized_2 = fused_features_flattened_2 / np.linalg.norm(fused_features_flattened_2)
    similarity_score = cosine_similarity([fused_features_normalized_1], [fused_features_normalized_2])[0][0]

    print(similarity_score)

    # Replace with your Gabor filter fitness evaluation logic
    # The goal is to maximize or minimize a certain criterion
    # For simplicity, we'll use a placeholder here (sum of parameters)
    
    return [1-similarity_score,filtered_finger_vein2,thresh2]


def honey_badger_optimize_gabor(num_honeybadgers, num_dimensions, search_space, image, generations):
    fitness_history = []  # To store fitness values over generations
    honeybadgers = [HoneyBadger(num_dimensions, search_space) for _ in range(num_honeybadgers)]

    for generation in range(generations):
        for honeybadger in honeybadgers:
            honeybadger.fitness = evaluate_gabor_fitness(honeybadger.position, image,similarity_score_list)[0]

            if honeybadger.fitness < evaluate_gabor_fitness(honeybadger.best_position, image,similarity_score_list)[0]:
                honeybadger.best_position = np.copy(honeybadger.position)
        
        fitness_history.append(max(honeybadgers, key=lambda x: x.fitness).fitness)
        
        # Perform adaptive updates or exploration based on honey badger behavior
        for honeybadger in honeybadgers:
            exploration_rate = 0.1
            exploration = np.random.uniform(-exploration_rate, exploration_rate, num_dimensions)
            honeybadger.position = honeybadger.best_position + exploration

            # Ensure the new position is within trhe search space
            honeybadger.position = np.clip(honeybadger.position, search_space[0], search_space[1])

    # Return the best Gabor filter parameters found
    global_best_honeybadger = max(honeybadgers, key=lambda x: x.fitness)
    return global_best_honeybadger.best_position,fitness_history

# Example usage
def apply_gabor_filter(image, theta=0, sigma=0.5, lambda_=2.0, gamma=0.5, phi=0):
   kernel_size = 31
   g_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_, gamma, phi, ktype=cv2.CV_32F)
   filtered_img = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)
   return filtered_img

finger_vein2 = cv2.imread('finger_vein2.bmp',cv2.IMREAD_GRAYSCALE)
finger_vein2 = cv2.resize(finger_vein2, (128, 128)) 

finger_vein3 = cv2.imread('index_6.bmp',cv2.IMREAD_GRAYSCALE)
finger_vein3 = cv2.resize(finger_vein3, (128, 128)) 


num_honeybadgers = 30
num_dimensions = 5  # Number of parameters in the Gabor filter
search_space = (0, 1)  # Example search space for Gabor filter parameters
generations = 50
similarity_score_list = []
clahe = cv2.createCLAHE(clipLimit=5)
finger_vein3 = clahe.apply(finger_vein3) + 25

best_gabor_parameters,fitness_history = honey_badger_optimize_gabor(num_honeybadgers, num_dimensions, search_space, finger_vein3, generations)
values = evaluate_gabor_fitness(best_gabor_parameters, finger_vein3,similarity_score_list)
fitness,thresh1,thresh2 = values[0],values[1],values[2]

print("Best Gabor Parameters:", best_gabor_parameters)
print("Fitness Value:", fitness)
kernel = cv2.getGaborKernel((31, 31), 0.19300665,0.18079031,0.06188156,0.214496,0.80236231, ktype=cv2.CV_32F)
filtered_image = cv2.filter2D(finger_vein3, cv2.CV_8U, kernel)

img_global = exposure.equalize_hist(filtered_image)

kernel = morp.disk(0)
img_local = rank.equalize(img_global, selem=kernel)



# r = cv2.selectROI("select the area", img_local) 
  
# Crop image 
# cropped_image = img_local[int(r[1]):int(r[1]+r[3]),  
#                       int(r[0]):int(r[0]+r[2])] 
# cropped_image = cv2.resize(cropped_image, (138, 128)) 
cropped_image = cv2.medianBlur(img_local,5)
thresh1 = cv2.adaptiveThreshold(cropped_image, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 199, 5)
figure, axis = plt.subplots(2, 2) 

axis[0, 0].imshow(finger_vein3,cmap = 'gray') 
axis[0, 1].imshow(filtered_image,cmap ='gray') 
axis[1, 0].imshow(cropped_image,cmap ='gray') 
axis[1, 1].imshow(thresh1,cmap ='gray') 


plt.show()