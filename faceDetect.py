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
        
        return face

    return None

if __name__ == "__main__":
    # Pulls the NN from deeplearner.py and switches to eval mode
    model = dl.SiameseNetwork()
    model.eval()

    # Gets the length of comparison directory
    len_directory = len(os.listdir("pictures"))
    directory = os.listdir("pictures")
    print(directory, len_directory)

    # Loop to compare all images (No working algorithm yet)
    for i in range(0, len_directory - 1):
        for j in range(i, len_directory - 1):
            path = os.path.join("pictures", directory[i])
            path2 = os.path.join("pictures", directory[j])

            img = cv.imread(path)
            img2 = cv.imread(path2)

            crop = recognize_face(img)
            crop2 = recognize_face(img2)

            if not crop.any() == None and not crop2.any() == None:
                crop_tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float()  
                crop2_tensor = torch.from_numpy(crop2).unsqueeze(0).unsqueeze(0).float()  

                out, out2 = model(crop_tensor, crop2_tensor)

                loss = F.pairwise_distance(out, out2)

                print("Difference: ", loss.item())
            else:
                print("No Face found for one of the images")
