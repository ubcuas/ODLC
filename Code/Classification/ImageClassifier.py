# NOTE: cv2 and pytesseract can take several minutes to install
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pytesseract
import imutils
from PIL import Image
import binascii
import scipy
import scipy.misc
import scipy.cluster

"""
Image Classification that handles image character detection and classificiation
"""
class ImageClassifier:
    # Characters can only be so large in the large, so these pixel sizes
    #   count as an extra filter towards only finding the correct images
    MIN_PIXEL_SIZE_CHAR = 900
    MIN_PIXEL_SIZE_BLOCK = 2250
    MAX_PIXEL_SIZE_BLOCK = 16000

    # The mimimum confidence (out of 100) that the classifier needs to say 
    #   that a given contour actually has a letter
    MIN_CONF_AMOUNT = 40

    # Used to give each of the detected subImages a name for file saving and for what is printed
    Image_Counter = 0

    # Configs so it only detects a single character, and also which characters it can detect
    #   https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
    TESS_CONFIG = ("-c tessedit"
                    "_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    " --psm 10"
                    " --oem 2"
                    " -l osd"
                    " ")
    TESS_CONFIG_2 = ("-c tessedit"
                    "_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    " --psm 10")

    IMAGES_LOCATION = "Images/"

    ## Other class variables
    #IMAGE_NAME # "ToBeDetected/" + name_img
    #NAME # the actual name "TheImage3" without extention
    #MAIN_IMAGE_LOCATION  #IMAGES_LOCATION + IMAGE_NAME 
    #ORIGINAL_IMAGE # the image as an image variable

    def __init__(self, image_name, name):
        self.IMAGE_NAME = image_name
        self.NAME = name
        self.MAIN_IMAGE_LOCATION = self.IMAGES_LOCATION + self.IMAGE_NAME
        self.ORIGINAL_IMAGE = cv2.imread(self.MAIN_IMAGE_LOCATION)

        self.Image_Counter = 0

    """
    Some ways to get an image with a filter
       Easier to detect things if they are only in greyscale or otherwise in some
       edited format that highlights edges betters
    """

    def get_original_image(self):
        return self.ORIGINAL_IMAGE

    def get_image(self, locationWithEndName):
        return cv2.imread(self.IMAGES_LOCATION + locationWithEndName)

    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        return cv2.medianBlur(image,5)
    
    ## thresholding
    #   Makes an image only solid black or only solid white
    #   Min and max determines what color something has to be for it to be either color
    #REQUIREMENTS: 
    #   Image is Greyscale, 
    #   min = [0,224]
    #   max = [1,225] 
    #   min < max
    def thresholding(self, image, min = 0, max = 255):
        return cv2.threshold(image, min, max, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    #dilation
    def dilate(self, image, size = (3,3)):
        # Kernel size increases or decreases the area of detecting rectangle
        # A smaller value like (10, 10) will detect words
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        #rect_kernel = np.ones((5,5),np.uint8)

        return cv2.dilate(image, rect_kernel, iterations = 1)
        
    #erosion
    def erode(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    # !Important filtering method, look into it
    def get_canny(self, image, min = 200, max = 300):
        return cv2.Canny(image, min, max)

    #skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    #template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 
    

    """Helpers"""

    # RETURNS: the text it detects in the image based on the config
    #Intention is to be usd on a cropped image with a single character to detect what that char is
    #Default is to detect single alphanumeric characters
    def get_text_in_image(self, image, config_ = '--psm 10'):
        return pytesseract.image_to_string(image, lang='eng', config = config_)

    # RETURNS: the confidence (out of 100) that the image is that the char
    #   it detected is the correct one
    # ASSUMES: you want to find the value of the first char (expected to be used on 1 char)
    def get_confidence_int(self, image):
        textData = pytesseract.image_to_data(image, output_type='data.frame', config = self.TESS_CONFIG)
        textData = textData[textData.conf != -1]
        conf = textData.groupby(['block_num'])['conf'].mean()
        if conf.empty:
            return 0
        confInt = int(list(conf)[0])

        return confInt


    """The actual parts of the code"""


    """
    NEEDS: Image variable
    RETURNS: 
        Contours of the image to detect, 
        Also the filtered image it got those contours from
    FILES (Images): Adds Base Images in Detected/.
    """
    def get_contours_in_image(self, image, useThresh = False):

        # Get versions of the image with filters
        imgGray = self.get_grayscale(image)
        imgCanny = self.get_canny(image, min=200, max=300)
        imgThresh = self.thresholding(imgGray)

        # The 0 is so that it appears at the beginning of the folder
        detectedName = self.IMAGES_LOCATION + "Detected/" + self.NAME + "/0" 

        # If the image is the original Image then give it an extra part to the name
        if image.all() == self.ORIGINAL_IMAGE.all():
            detectedName = detectedName + "_OG"

        # Shows how the image is being filters to help debug and find ideal image
        cv2.imwrite(detectedName + '_img.png', image)
        cv2.imwrite(detectedName + '_canny.png', imgCanny)
        cv2.imwrite(detectedName + '_gray.png', imgGray)
        cv2.imwrite(detectedName + '_thresh.png', imgThresh)

        # Find Contours from the Threshholding or Canny
        selectedImg = imgThresh if useThresh else imgCanny

        imgDilate = self.dilate(selectedImg)
        contours, hierarchy = cv2.findContours(imgDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        return contours, selectedImg
    
    """
    NEEDS: Image variable
    FILES (Images): Detected/.
    
    From the contours, selects the ones it believes contains letters and puts it in Detected/.
    """
    def get_letters_from_contours(self, contours, image, findLargeBlocks = False):
        #Dont want to mess with Original Image
        imgCopy = self.ORIGINAL_IMAGE.copy()
        imgRectangles = self.ORIGINAL_IMAGE.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            #If you want to find larger blocks in a specific size range
            if (findLargeBlocks and (self.MIN_PIXEL_SIZE_BLOCK > w*h or w*h > self.MAX_PIXEL_SIZE_BLOCK)):
                continue

            #Should have an area of at least 900 pixels to detect 
            #900 was chosen as it is a 30x30 pixel img
            if (w * h) < self.MIN_PIXEL_SIZE_CHAR:
                continue
            
            # Drawing a rectangle on copied image
            # 0_imgRectangles should show the "characters" it detects
            rect = cv2.rectangle(imgRectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(self.IMAGES_LOCATION + "Detected/" + self.NAME + "/0" + '_imgRectangles.png', imgRectangles)
            
            # Cropping the image so it is only the detected Contour
            imgCropped = imgCopy[y:y + h, x:x + w]        
            
            # Apply OCR on the cropped image
            text = self.get_text_in_image(imgCropped, self.TESS_CONFIG_2)

            # If we are not confident about the detected text, then just skip
            confInt = self.get_confidence_int(imgCropped)
            if confInt < self.MIN_CONF_AMOUNT:
                continue
            confStr = str(confInt)

            # Save image to compare to the Detected Text
            cv2.imwrite(self.IMAGES_LOCATION + "Detected/" + self.NAME + "/" + str(self.Image_Counter) + ".png", imgCropped) 

            # Increase Counter so we don't have images with same file name
            self.Image_Counter += 1

    """
    NEEDS: Image variable
    FILES: 
        Images/Identified/.
        Tests/recognized.txt

    Rotates the image and applies final filters to actually identify which letter it detects in Detected
        Puts the rotated/filtered image in Identified and prints out detailed information on the images in Detected
    """
    def rotation_identifier(self, image, imgNum, rotations = 12, method = "Sharp"):

        #DetectionInfo
        rotatedImageCharConfidence = {}
        maxConfLetter = ["0", 0, 0] #Letter, Confidence, Rotation

        # Make rotations into factor of 360
        # Probably not necessary, but ig it keeps rotationDegrees an int
        rotations += 360 % rotations 

        #Color stuff, did not fully finish implementing before deciding we won't do this
        #   So commented out, but techically "works" and shouldn't crash you
        #print(self.hist_curve(image))
        #self.colors(image)

        # Rotate the image by increments of how many rotations you wanted
        #   To see if it detects letters better in some other orientation
        for i in range(rotations):
            rotationDegree = int(i * (360 / rotations))

            imgRotated = imutils.rotate_bound(image, rotationDegree)

            #Should grayscale be applied to the method to help in classification? (and before or after?)
            imgRotated = self.get_grayscale(imgRotated)
            imgRotated = self.get_image_from_method(imgRotated, method)
            #imgRotated = self.get_grayscale(imgRotated)

            #First letter, in lowercase (competition doesnt ask for capitalization)
            local_config = " --psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 page_separator=''"
            rotatedText = str(self.get_text_in_image(imgRotated, local_config).lower()[0]).strip() 
            rotatedConfidence = self.get_confidence_int(imgRotated)

            #If the text it detects is not empty, then add its info to the DetectionInfo
            if rotatedText != "":
                if rotatedText in rotatedImageCharConfidence:
                    rotatedImageCharConfidence[rotatedText] += rotatedConfidence
                    if (rotatedConfidence > maxConfLetter[1]):
                        maxConfLetter = [rotatedText, rotatedConfidence, rotationDegree]
                else:
                    rotatedImageCharConfidence[rotatedText] = rotatedConfidence

                if (rotatedConfidence > maxConfLetter[1]):
                    maxConfLetter = [rotatedText, rotatedConfidence, rotationDegree]

            #File names will be:   (#ofImageInDetected)_(rotationDegrees)_(MethodOfClassification).png
            cv2.imwrite(self.IMAGES_LOCATION + "Identified/" + self.NAME + "/"+ imgNum + "_" + str(rotationDegree) + "_" + method +".png", imgRotated)

        #Printing out the info, 
        #   the dictionary it prints tells you the summed confidence of each letter
        #   whether to choose the letter with the single highest confidence or letter with
        #   the highest summed confidence is up to you
        print("Relevant Detected Image Number: " + imgNum)
        print("Best Letter: " + maxConfLetter[0] + " Confidence: " + str(maxConfLetter[1]) + " Rotation: " + str(maxConfLetter[2]))
        modal_key, modal_val = self.max_val_from_dict(rotatedImageCharConfidence)
        print("Modal Confident Letter: " + str(modal_key) + " Summed Confidence: " + str(modal_val))
        print(rotatedImageCharConfidence)
        print()

        # Appending the detected text into file
        file = open("Code/Tests/recognized.txt", "a")  
        file.write("Img " + imgNum + ": " + maxConfLetter[0] + ", " + str(modal_key) + "\n")
        file.close()

    # Helper for function above
    # NEEDS: dictionary in the form of {string:int}
    # RETURNS: max value from a dict
    def max_val_from_dict(self, dict):
        maxval = 0
        maxkey = ""

        for key in dict:
            if dict[key] > maxval:
                maxval = dict[key]
                maxkey = key

        return maxkey, maxval

    #Other helper for rotation one
    # NEEDS: Image and what method you want to use
    #only sharpen exists for the moment
    #sharpen was chosen because it accentuates the edges well
    def get_image_from_method(self, image, method):
        if (method == "None"):
            return image
        if (method == "Sharp"):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(image, -1, kernel)


    """
    from: https://github.com/opencv/opencv/blob/master/samples/python/hist.py
    Was looking into it for the color thing, but didn't fully implent so no full use rn
    """
    def hist_curve(self, im):
        bins = np.arange(256).reshape(256,1)
        h = np.zeros((300,256,3))
        if len(im.shape) == 2:
            color = [(255,255,255)]
        elif im.shape[2] == 3:
            color = [ (255,0,0),(0,255,0),(0,0,255) ]
        for ch, col in enumerate(color):
            hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
            cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
            hist=np.int32(np.around(hist_item))
            pts = np.int32(np.column_stack((bins,hist)))
            cv2.polylines(h,[pts],False,col)
        y=np.flipud(h)
        return y

    """
    Two ways I intended this to be used:
    1) The actual color detection for the UAS image detection requirements
    2) Color filtering for more accurate letter detection:
        i) Read for info
            https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
        a) Basically find out the most common color in the image, this would be the background
        b) Then filter out that color so hopefully only the letter is left

    # GETS: Image
    # RETURNS: Main color of that image
    """
    def colors(self, image):
        NUM_CLUSTERS = 10

        #image = image.resize((150, 150))      # optional, to reduce time
        ar = np.asarray(image)
        try:
            shape = ar.shape
            ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        except:
            pass

        #print('finding clusters')
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        #print('cluster centres:\n', codes)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

        index_max = scipy.argmax(counts)                    # find most frequent
        peak = codes[index_max]
        colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
        print('most frequent color is %s (#%s)' % (peak, colour))





