# Imports OpenCV and Matplotlib's pyplot
import cv2 as cv
import deeplearner as dl
import torch
import torch.nn.functional as F
import os

# Detects faces and crops images
def recognize_face(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initializes the face class for face detection
    face_class = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml" # type: ignore
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

    model = dl.SiameseNetwork()
    model.load_state_dict(torch.load("state.pt"))
    model.eval()

    img = cv.imread("pictures/mahogany.jpg")
    img2 = cv.imread("pictures/majd.jpg")

    crop = recognize_face(img)
    crop2 = recognize_face(img2)

    crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0)
    crop2_tensor = torch.tensor(crop2, dtype=torch.float32).unsqueeze(0)

    out, out2 = model(crop_tensor, crop2_tensor)

    loss = F.pairwise_distance(out, out2).mean()

    print(loss)
