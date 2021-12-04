# Image Classifier

The role of this class is to recieve an image and return images where it has classified an alphanumeric character alongside its color as specified in the UAS rules

## Use
1) Get the image
- imageClassifier = IC("ToBeDetected/" + name_img, name)
2) Use that image to find contours (edge detection)
- image = imageClassifier.get_original_image()
- contours, selectedImg = imageClassifier.get_contours_in_image(image)
3) Loop through the contours and search for the ones that are likely to contain letters
- imageClassifier.get_letters_from_contours(contours, image, 0)

4) Loop through all the contours you found with letters and apply transformations to find the specific letter and color
- for img_i in range(imageClassifier.Image_Counter):
-     imgDetected = imageClassifier.get_image("Detected/" + name + "/" + str(img_i) + ".png")
-     imageClassifier.rotation_identifier(imgDetected, str(img_i))


## Constants
- Can be changed as needed
MIN_PIXEL_SIZE_CHAR = 900
MIN_PIXEL_SIZE_BLOCK = 2250
MAX_PIXEL_SIZE_BLOCK = 16000
MIN_CONF_AMOUNT = 40

## Functions
* get_text_in_image(image, config):
  * Image, *string*

Given an image, it returns the text it finds, default config returns alphanumeric character.

* get_confidence_int(image)
  * Image

Gets text in the same exact way as get_text_in_image with default config, but returns how confident it was with the result

* get_contours_in_image(image, useThresh):
  * Image, *boolean*

Given an image, it applies cropping and filtering methods before using cv2.findContours to get contours. Also inputs intermediate images in the Images/Detected/. folder

* get_letters_from_contours(contours, image, findLargeBlocks):
  * Contours, image, *boolean*

Loops through all the contour and based on the image size and confidence, returns the ones that it thinks has letters

* rotation_identifier(image, imgNum, rotations, method)
  * Image, int, int, *string*

Given an image and it's number (the names of identified images in Detected are in the forms of number), it will rotate the image 'rotations' times and apply the 'method's filter to find which letter it detects the most or with most confidence.