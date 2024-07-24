import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# function for grayscale and color
def read_image_grayscale(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def read_image_color(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def display_images(images, titles, cmap=None, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, len(images) // 2, i)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
    plt.tight_layout()
    plt.show()

# Load images
moon_img = read_image_color('moon.jpg')
morph_img = read_image_grayscale('morphology.png')
fingerprint_img = read_image_grayscale('fingerprint_BW.png')
cell_img = read_image_grayscale('cell.jpg')
cell_img_color = read_image_color('cell.jpg')

# Apply lapacian and sobel for Part 1 
laplacian = cv2.Laplacian(moon_img, cv2.CV_64F)
sobelx = cv2.Sobel(moon_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(moon_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.bitwise_or(sobelx, sobely).astype(np.uint8)
#Apply dilated opening and closed 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated_img = cv2.dilate(morph_img, kernel, iterations=1)
opened_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel_close)
closed_dilated_img = cv2.dilate(closed_img, kernel, iterations=1)  # Reusing the initial kernel
# difference between median and morph
median_filtered = cv2.medianBlur(fingerprint_img, 5)
morph_filtered = cv2.morphologyEx(median_filtered, cv2.MORPH_CLOSE, kernel_close)
# cell calculation, biggest cell 
_, bw_img = cv2.threshold(cell_img, 127, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_img, 8, cv2.CV_32S)
cell_img_color_rgb = cv2.cvtColor(cell_img_color, cv2.COLOR_BGR2RGB) 
cell_sizes = stats[1:, cv2.CC_STAT_AREA]
cell_data = pd.DataFrame({
    'Cell Label': np.arange(1, num_labels),
    'Size in pixels':cell_sizes
})

# Display and plotting code remains largely unchanged, focusing on performance issues in image processing
plt.figure(figsize=(10, 10))
plt.subplot(2, 3, 1), plt.imshow(moon_img, cmap='gray'), plt.title('Original Moon')
plt.subplot(2, 3, 2), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.subplot(2, 3, 5), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.subplot(2, 3,  3), plt.imshow(sobelx , cmap= 'gray'), plt.title('Sobel X')
plt.subplot(2, 3,  4), plt.imshow(sobely , cmap= 'gray'), plt.title('Sobel Y')

plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1), plt.imshow(morph_img, cmap='gray'), plt.title('Original Morphology')
plt.subplot(2, 2, 2), plt.imshow(opened_img, cmap='gray'), plt.title('Opened')
plt.subplot(2, 2, 3), plt.imshow(closed_img, cmap='gray'), plt.title('Closed')
plt.subplot(2, 2, 4), plt.imshow(closed_dilated_img, cmap='gray'), plt.title('Closed & Dilated')
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1), plt.imshow(fingerprint_img, cmap='gray'), plt.title('Original Fingerprint')
plt.subplot(2, 2, 2), plt.imshow(median_filtered, cmap='gray'), plt.title('Median Filtered')
plt.subplot(2, 2, 3), plt.imshow(morph_filtered, cmap='gray'), plt.title('Morphological Filtered')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1), plt.imshow(cell_img, cmap='gray'), plt.title('Original Cell')
plt.subplot(2, 2, 2), plt.imshow(labels, cmap='nipy_spectral'), plt.title('Labelled Cells')
plt.subplot(2, 2, 3), plt.imshow(cell_img_color_rgb), plt.title('Colored Cells')
plt.tight_layout()
plt.show()


# Print cell count and size of the biggest cell
print(f'Total number of cells: {num_labels - 1}')
biggest_cell_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
print(f'Size of the biggest cell: {stats[biggest_cell_index, cv2.CC_STAT_AREA]} pixels')
x ,y, w, h, area = stats[biggest_cell_index, cv2.CC_STAT_LEFT] , stats[biggest_cell_index, cv2.CC_STAT_TOP] , stats[biggest_cell_index, cv2.CC_STAT_WIDTH] , stats[biggest_cell_index, cv2.CC_STAT_HEIGHT], stats[biggest_cell_index, cv2.CC_STAT_AREA]
# Corrected section for highlighting the biggest cell
cv2.rectangle(cell_img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw on the colored image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(cell_img_color, cv2.COLOR_BGR2RGB))  # Correct color representation for display
plt.title('Biggest Cell Highlighted')
plt.show()
print(cell_data)


# Print cell count and size of the biggest cell