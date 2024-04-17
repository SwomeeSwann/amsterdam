# Imports OpenCV and Matplotlib's pyplot
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import PIL as pil

# Detects faces and crops images
def recognize_face(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initializes the face class for face detection
    face_class = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detects faces in the image
    picFace = face_class.detectMultiScale(
        gray_img, scaleFactor=1.2593, minNeighbors=4, minSize=(50,50)
    )

    # Creates rectangles around faces detected
    for (x, y, w, h) in picFace:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Transforms the image into an rgb one for matplotlib
    print(picFace)

    # return the picture of the face, the rgb image (for debugging), and the gray image for matplotlib
    return picFace

img = cv.imread("mahogany.jpg")
print(img)
recognize_face(img)