from Classification.ImageClassifier import ImageClassifier as IC
from Tests.Tests import Tests
import os

"""
Given: The name of an image in Images/ToBeDetected (with 3 letter extension)

1) Creates a directory in Images/Detected where it stores Letters it 
    supposedly detects from contours
2) Rotates and applies filters to these cropped detected image to figure out 
    the character it detects alongside orientation and color
"""
if __name__ == '__main__':
    ## Startup
    print('Started')
    name_img = "Letters Med_Long.jpg"
    name = name_img[:-4] #removes .png, .jpg or other 3 letter at back

    if not os.path.exists(os.path.join(os.getcwd(), "Images/Detected/" + name)):
        os.mkdir(os.path.join(os.getcwd(), "Images/Detected/" + name))
    if not os.path.exists(os.path.join(os.getcwd(), "Images/Identified/" + name)):
        os.mkdir(os.path.join(os.getcwd(), "Images/Identified/" + name))

    ## Detection Steps
    imageClassifier = IC("ToBeDetected/" + name_img, name)
    image = imageClassifier.get_original_image()
    contours, selectedImg = imageClassifier.get_contours_in_image(image)
    imageClassifier.get_letters_from_contours(contours, image, 0)

    #Empty the txt, for testing purposes
    open('Code/Tests/recognized.txt', 'w').close()

    ## Identification Steps
    for img_i in range(imageClassifier.Image_Counter):
        imgDetected = imageClassifier.get_image("Detected/" + name + "/" + str(img_i) + ".png")
        imageClassifier.rotation_identifier(imgDetected, str(img_i))
    print('Finished')

    tests = Tests()
    tests.highest_accuracy()
    tests.modal_accuracy()

