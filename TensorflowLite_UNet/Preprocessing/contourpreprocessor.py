import imutils
import cv2

class ContourPreprocessor:
    def __init__(self, inverted=True):
    # select inverted image
        self.inverted = inverted

    def preprocess(self, image):
    # grab the dimensions of the image and then initialize
    # the deltas to use when cropping
        (h, w) = image.shape[:2]
        #image_new = image.copy();

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #thresh_inv = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

        thresh_inv = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)[1]

        #thresh_inv = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        (x, y, w, h) = cv2.boundingRect(cnt[0])
        if self.inverted:
            ret_image = cv2.merge([thresh_inv[y:y+h, x:x+w]]*3)
        else:
            thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
            ret_image = cv2.merge([thresh[y:y+h, x:x+w]] * 3)
        return ret_image
