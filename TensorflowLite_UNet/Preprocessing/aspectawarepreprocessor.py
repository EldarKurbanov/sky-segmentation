import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_NEAREST, contours=False, pad=0):
    # store the target image width, height, and interpolation
    # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        self.contours = contours;
        self.pad = pad;

    def preprocess(self, image):
    # grab the dimensions of the image and then initialize
    # the deltas to use when cropping
        (h, w) = image.shape[:2]
        #image_new = image.copy();


        if w > h:
            image = imutils.resize(image, width=self.width)
        else:
            image = imutils.resize(image, height=self.height)



        # obtain the target dimensions
        padW = int((self.width - image.shape[1]) / 2.0)
        padH = int((self.height - image.shape[0]) / 2.0)


        if self.contours:
            #image = cv2.copyMakeBorder(image, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT, value=0)
            image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            #image = cv2.copyMakeBorder(image, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT, value=255)
            image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            #image = cv2.copyMakeBorder(image, 0, self.pad, 0, 0, cv2.BORDER_CONSTANT, value=255)



        if self.contours:
            if image.shape[1] < self.width:
                image = cv2.copyMakeBorder(image, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if image.shape[0] < self.height:
                image = cv2.copyMakeBorder(image, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image = cv2.copyMakeBorder(image, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            if image.shape[1] < self.width:
                image = cv2.copyMakeBorder(image, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if image.shape[0] < self.height:
                image = cv2.copyMakeBorder(image, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            image = cv2.copyMakeBorder(image, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # finally, resize the image to the provided spatial
        # dimensions to ensure our output image is always a fixed
        # size
        #return image
        return cv2.resize(image, (self.width, self.height))