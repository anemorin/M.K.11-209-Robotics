import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    """Load an image from the specified path."""
    return cv2.imread(path)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_sharpening(image, method='kernel'):
    """Apply sharpening using either kernel or unsharp mask method."""
    if method == 'kernel':
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    else:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

def detect_edges(image, ksize=5):
    """Detect edges using Sobel operator."""
    sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=ksize)
    return cv2.convertScaleAbs(sobel)

def combine_images(blurred, edges, sharpened):
    """Combine processed images with weighted addition."""
    combined = cv2.addWeighted(blurred, 0.5, edges, 0.5, 0)
    return cv2.addWeighted(combined, 0.5, sharpened, 0.5, 0)

def show_images(original, blurred, edges, sharpened, combined):
    """Display all processed images using matplotlib."""
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ')
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости')
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Комбинация изображений')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    image_path = "C:\KFU\M.K.11-209-Robotics\Lab1\image.jpg"
    image = load_image(image_path)

    if image is None:
        print("Ошибка: Не удалось загрузить изображение")
        return

    blurred = apply_gaussian_blur(image)
    sharpened = apply_sharpening(image, method='kernel')
    edges = detect_edges(image)
    combined = combine_images(blurred, edges, sharpened)

    show_images(image, blurred, edges, sharpened, combined)

if __name__ == "__main__":
    main()