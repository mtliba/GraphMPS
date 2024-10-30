import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class GaussianCenterBias:
    def __init__(self, height, width, sigma_x=0.2, sigma_y=0.4, base_fixations=200, weight_factor=2):
        self.height = height
        self.width = width
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.base_fixations = base_fixations
        self.weight_factor = weight_factor
        
    def create_vertical_gaussian_center_bias_map(self):
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Calculate the Gaussian with different sigma values for x and y
        bias_map = np.exp(-(xv**2 / (2 * self.sigma_x**2) + yv**2 / (2 * self.sigma_y**2)))

        # Normalize to [0, 1] range
        bias_map = (bias_map - bias_map.min()) / (bias_map.max() - bias_map.min())

        return bias_map
    
    def get_high_gaussian_mask(self, gaussian_map, percentile=80):
        gaussian_map_normalized = cv2.normalize(gaussian_map, None, 0, 255, cv2.NORM_MINMAX)
        threshold = np.percentile(gaussian_map_normalized, percentile)
        high_gaussian_mask = (gaussian_map_normalized >= threshold).astype(np.uint8)
        return high_gaussian_mask, gaussian_map_normalized
    
    def calculate_fixations(self, high_gaussian_mask):
        salient_area_ratio = np.sum(high_gaussian_mask) / high_gaussian_mask.size
        num_fixations = int(self.base_fixations * salient_area_ratio)
        return num_fixations
    
    def apply_weighted_sampling(self, gaussian_map, high_gaussian_mask, num_fixations):
        high_gaussian_map = gaussian_map * high_gaussian_mask
        weighted_gaussian_map = high_gaussian_map ** self.weight_factor
        weighted_gaussian_map_prob = weighted_gaussian_map / np.sum(weighted_gaussian_map)
        smooth_weighted_gaussian_map_prob = gaussian_filter(weighted_gaussian_map_prob, sigma=5)
        smooth_weighted_gaussian_map_prob /= np.sum(smooth_weighted_gaussian_map_prob)
        
        flat_indices = np.random.choice(
            smooth_weighted_gaussian_map_prob.size, size=num_fixations, p=smooth_weighted_gaussian_map_prob.ravel()
        )
        fixation_y, fixation_x = np.unravel_index(flat_indices, high_gaussian_map.shape)
        return fixation_y, fixation_x, high_gaussian_map

    def plot_fixations(self, high_gaussian_map, fixation_x, fixation_y):
        plt.figure(figsize=(12, 6))
        plt.imshow(high_gaussian_map, cmap='gray')
        plt.scatter(fixation_x, fixation_y, color='red', s=10, alpha=0.8, label='Fixations')
        plt.title(f'Fixation Distribution (Using Gaussian Center Bias)')
        plt.axis('off')
        plt.legend(loc='upper right')
        plt.show()

    def generate_fixations(self):
        gaussian_map = self.create_vertical_gaussian_center_bias_map()
        high_gaussian_mask, gaussian_map_normalized = self.get_high_gaussian_mask(gaussian_map)
        num_fixations = self.calculate_fixations(high_gaussian_mask)
        fixation_y, fixation_x, high_gaussian_map = self.apply_weighted_sampling(gaussian_map_normalized, high_gaussian_mask, num_fixations)
        self.plot_fixations(high_gaussian_map, fixation_x, fixation_y)
        
if __name__ == "__main__":
    height, width = 200, 60
    sigma_x, sigma_y = 0.25, 0.7
    base_fixations = 200
    weight_factor = 2
    gaussian_center_bias = GaussianCenterBias(height, width, sigma_x, sigma_y, base_fixations, weight_factor)
    gaussian_center_bias.generate_fixations()

