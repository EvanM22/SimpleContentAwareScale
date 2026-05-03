import cv2
import numpy as np
from scipy.ndimage import convolve
from numba import njit

def calculate_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Scharr is often more accurate for edges than standard Sobel
    grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy

@njit
def find_vertical_seam(energy):
    rows, cols = energy.shape
    # energy is float64, so dist should be too
    dist = energy.copy()
    # Explicitly use np.int32 for Numba compatibility
    backtrack = np.zeros((rows, cols), dtype=np.int32)

    for r in range(1, rows):
        for c in range(cols):
            # Define the search range for the previous row
            left = max(0, c - 1)
            right = min(cols, c + 2)
            
            # Find the index of the minimum energy path above
            # We use a manual loop or specialized slice for Numba
            prev_row_slice = dist[r - 1, left:right]
            
            # Manual argmin is often faster in Numba than np.argmin on small slices
            min_val = prev_row_slice[0]
            min_idx = 0
            for i in range(1, len(prev_row_slice)):
                if prev_row_slice[i] < min_val:
                    min_val = prev_row_slice[i]
                    min_idx = i
            
            backtrack[r, c] = left + min_idx
            dist[r, c] += min_val

    # Backtrack to find the actual path
    seam = np.zeros(rows, dtype=np.int32)
    
    # Find min in the last row
    last_row = dist[-1, :]
    min_val_last = last_row[0]
    curr_idx = 0
    for i in range(1, len(last_row)):
        if last_row[i] < min_val_last:
            min_val_last = last_row[i]
            curr_idx = i
            
    seam[-1] = curr_idx
    
    for r in range(rows - 2, -1, -1):
        seam[r] = backtrack[r + 1, seam[r + 1]]
    
    return seam

def remove_vertical_seam(img, seam):
    rows, cols, ch = img.shape
    # Create a boolean mask of pixels to keep
    mask = np.ones((rows, cols), dtype=bool)
    rows_idx = np.arange(rows)
    mask[rows_idx, seam] = False
    
    # Reshape mask to apply across all 3 color channels
    return img[mask].reshape(rows, cols - 1, 3)

def seam_carve(img, iterations):
    """
    Iteratively removes seams to shrink the image width.
    """
    for i in range(iterations):
        print(f"Carving seam {i+1}/{iterations}...")
        energy = calculate_energy(img)
        seam = find_vertical_seam(energy)
        img = remove_vertical_seam(img, seam)
    return img

# --- Execution ---
input_path = 'C:\perishcaptionfix.png' # Support for .bmp, .jpg, .png
image = cv2.imread(input_path)

if image is None:
    print("Error: Could not load image.")
else:
    # Scale width down by 50 pixels
    num_seams_to_remove = 500 
    result = seam_carve(image, num_seams_to_remove)

    cv2.imshow("Original", image)
    cv2.imshow("Content-Aware Scaled", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
