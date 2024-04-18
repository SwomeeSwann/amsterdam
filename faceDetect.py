# Imports OpenCV and Matplotlib's pyplot
import cv2 as cv
import deeplearner as dl
import torch
import torch.nn.functional as F

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

        face = gray_img[y:y+h, x:x+h]

        face = cv.resize(face, (100, 100))
    # Transforms the image into an rgb one for matplotlib

    # return the picture of the face, the rgb image (for debugging), and the gray image for matplotlib
    return face

if __name__ == "__main__":
    img = cv.imread("mahogany.jpg")
    img2 = cv.imread("majd.jpg")

    crop = recognize_face(img)
    crop2 = recognize_face(img2)

    crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()
    crop_tensor2 = torch.from_numpy(crop2).unsqueeze(0).float()

    result = dl.network.forward_once(crop_tensor)
    result2 = dl.network.forward_once(crop_tensor2)

    loss = F.pairwise_distance(result, result2)

    print(loss[0])
