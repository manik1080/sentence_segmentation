import cv2
import os
import pickle


class Process:
    def __init__(self, image_directory, show):
        self.working_directory = image_directory
        self.show = show


    def create_image_array(self, path):
        data_list = []
        img = cv2.imread(os.path.join(self.working_directory, path))
        #contrast, brightness = (1.4, 1.4)
        #img = cv2.addWeighted(img, contrast, img, 0, brightness)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        otsu1 = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

        horz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        lines = cv2.morphologyEx(otsu1, cv2.MORPH_OPEN, horz_kernel, iterations=1)
        cnts = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(img, [c], -1, (255, 255, 255), 2)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        otsu2 = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 6))
        dilation = cv2.dilate(otsu2, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        count = 0
        for j, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            if w > 90 and h > 20:
                count += 1
                out = cv2.resize(otsu2[y:h + y, x:w + x], (256, 64))
                data_list.append(out)
                rect = cv2.rectangle(otsu2, (x, y), (x + w, y + h), (255, 255, 255), 2)
        if self.show:
            cv2.imshow(path, otsu2)
            cv2.waitKey(0)
        return data_list


    def equify(self, array):
        arr = [len(i) for i in array]
        MAX = max(arr)
        for sub in array:
            x = len(sub)
            count = 0
            while len(sub) < MAX:
                sub.append(sub[count % x])
                count += 1
        return array


    def run(self):
        image_data = []
        for img_path in os.listdir(self.working_directory):
            image_data.append(self.create_image_array(os.path.join(self.working_directory, img_path)))
        return self.equify(image_data)


class Extractor:
    def __init__(self, image_directory, show=False):
        self.working_directory = image_directory
        self.processing = Process(image_directory, show)


    def extract(self):
        data = self.processing.run()
        return data
    
    def extract_train_test(self, train_size=0.7, test_size=0.3):
        split = train_size + test_size
        data = self.processing.run()
        size = len(data)
        return (data[:int(train_size/split * size)], data[int(test_size/split * size):])


    def create_files(self, split=0.7):
        data = self.processing.run()
        size = len(data)
        with open(os.path.join(os.path.dirname(self.working_directory), 'train_data.bin'), 'wb') as f:
            pickle.dump(data[:int(split * size)], f)
        with open(os.path.join(os.path.dirname(self.working_directory), 'test_data.bin'), 'wb') as f:
            pickle.dump(data[int(split * size):], f)
