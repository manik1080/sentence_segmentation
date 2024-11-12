import pickle
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_shape=(512, 512), show=False, verbose=0):
        self.show = show
        self.verbose = verbose
        self.outputs = None
        self.image_height, self.image_width = image_shape

    def image_to_array(self, image_path):
        img = cv2.imread(image_path)
        return img

    def process_directory(self, image_directory):
        self.image_directory = image_directory
        data_arrays = []
        for filename in os.listdir(self.image_directory):
            image_path = os.path.join(self.image_directory, filename)
            img_array = self.image_to_array(image_path)
            if img_array is not None:
                self.resize(img_array, self.image_height, self.image_width)
                processed_image = self.extract(img_array)
                if processed_image is not None:
                    processed_image = self.to_numpy(processed_image)
                    data_arrays.append(processed_image)
            else:
                if self.verbose:
                    print(f"{filename} ignored: invalid or unreadable file.")
        return data_arrays

    def process_image(self, image_path):
        img_array = self.image_to_array(image_path)
        
        if img_array is not None:
            resized_image = self.resize_image(img_array, self.image_height, self.image_width)
            data_arrays = self.extract(resized_image)
            return data_arrays
        else:
            raise FileNotFoundError(f"Problem while reading image at {image_path}: invalid or unreadable file.")

    def to_numpy(self, array):
        arr = [len(i) for i in array]
        MAX = max(arr)
        for sub in array:
            x = len(sub)
            count = 0
            while len(sub) < MAX:
                sub.append(sub[count % x])
                count += 1
        return np.array(array)

    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height))

    def extract(self, img):
        result_img = img.copy()
        lines_removed = img.copy()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        otsu1 = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

        horz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
        lines = cv2.morphologyEx(otsu1, cv2.MORPH_OPEN, horz_kernel, iterations=2)
        cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            cv2.drawContours(lines_removed, [c], -1, (255, 255, 255), 2)

        grey = cv2.cvtColor(lines_removed, cv2.COLOR_BGR2GRAY)
        otsu2 = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 6))
        dilation = cv2.dilate(otsu2, kernel, iterations=1)
        # opening
        dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
                                  )

        img_list = []
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for idx, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            X, Y = img.shape[:-1]
            if w>X//23 and Y//4>h>Y//32:
                img_list.append(self.resize_image(otsu2[y:y + h, x:x + w], 128, 64))
                if self.show:
                    rect = cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.show:
            self.outputs = img, otsu1, otsu2, dilation, result_img
            self.display_results()

        return img_list

    def display_results(self):
        fig, ax = plt.subplots(2, 3, figsize=(12, 7))
        image, otsu1, otsu2, dilation, result = self.outputs
        ax[0, 0].imshow(image, cmap='gray')
        ax[0, 0].set_title('Image')
        ax[0, 1].imshow(otsu1, cmap='gray')
        ax[0, 1].set_title('Otsuing')
        ax[0, 2].imshow(image, cmap='gray')
        ax[0, 2].set_title('Lines Removed')
        ax[1, 0].imshow(otsu2, cmap='gray')
        ax[1, 0].set_title('Re-Otsuing')
        ax[1, 1].imshow(dilation, cmap='gray')
        ax[1, 1].set_title('Dilated')
        ax[1, 2].imshow(result, cmap='gray')
        ax[1, 2].set_title('Result')
        plt.show()

    def get_train_test(self, split=0.7):
        data = self.process_directory()
        split_idx = int(len(data) * split)
        return data[:split_idx], data[split_idx:]

    def make_files(self, train_path='train.bin', test_path='test.bin', split=0.7):
        train_data, test_data = self.get_train_test(split)
        with open(train_path, 'wb') as train_file:
            pickle.dump(train_data, train_file)
        with open(test_path, 'wb') as test_file:
            pickle.dump(test_data, test_file)


if __name__ == '__main__':
    path = "images/7.jpg"
    extractor = ImageProcessor(
                image_shape=(512, 512),
                show=True,
                verbose=True)
    words = extractor.process_image(path)
    print('(', end='')
    print(len(words), str(words[0].shape)[1:], sep=', ')
    print(extractor.to_numpy(words).shape)
