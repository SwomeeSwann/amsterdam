import cv2 as cv
import matplotlib.pyplot as plt


def recognize_face(img):
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_class = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    picFace = face_class.detectMultiScale(
        grey_img, scaleFactor=1.2593, minNeighbors=4, minSize=(50,50)
    )

    for (x, y, w, h) in picFace:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)


    img_rgb =cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return picFace, img_rgb, grey_img


def compare_face(img1, img2):
    face1, rgb_img1, grey_img1 = recognize_face(img1)
    face2, rgb_img2, grey_img2 = recognize_face(img2)

    if len(face1) == 0 or len(face2) == 0:
        return False

    (x1, y1, w1, h1) = face1[0]
    (x2, y2, w2, h2) = face2[0]


    face_img1 = rgb_img1[y1:y1+h1, x1:x1+w1]
    face_img2 = rgb_img2[y2:y2+h2, x2:x2+w2]

    plt.figure(figsize=(20,10))
    plt.imshow(face_img1)
    plt.axis('off')
    
    plt.figure(figsize=(20,10))
    plt.imshow(face_img2)
    plt.axis('off')

    plt.show()

    hist1 = cv.calcHist([grey_img1], [0], None, [256], [0, 256])
    hist2 = cv.calcHist([grey_img2], [0], None, [256], [0, 256])

    similarity = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)

    threshold = 1000

    if similarity < threshold:
        print(similarity)
        return True
    else:
        print(similarity)
        return False



imagePath1 = 'mahogany.jpg'
imagePath2 = 'bruh.jpg'

img1 = cv.imread(imagePath1)
img2 = cv.imread(imagePath2)

if compare_face(img1, img2):
    print("same person")
else:
    print("different")



