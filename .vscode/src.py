#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Bruce Willis", "Matt Damon", "Jackie Chan", "Leonardo DiCaprio", "Denzel Washington", "Liv Tyler"]


#-------------------------------------------------------------------------------

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]


    return gray[y:y+w, x:x+h], faces[0]

#----------------------------------------------------------------------------------

def prepare_training_data(data_folder_path):
 
    dirs = os.listdir(data_folder_path)
    
    faces = [] 
    labels = []

    for dir_name in dirs:
 
        if not dir_name.startswith("s"):
            continue;
 
        label = int(dir_name.replace("s", ""))
        
     
        
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
 

        for image_name in subject_images_names:
       
            if image_name.startswith("."):
                continue;
 
            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path) 
            #cv2.imshow("Training on image...", image)
            #cv2.waitKey(100)
 
            face, rect = detect_face(image)
 

            if face is not None:
                faces.append(face)
                labels.append(label)
 
           # cv2.destroyAllWindows()
           # cv2.waitKey(1)
           # cv2.destroyAllWindows()
 
    return faces, labels

#-----------------------------------------------------------------------------------


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createEigenFaceRecognizer()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createFisherFaceRecognizer()


#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))




def draw_rectangle(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 #---------------------------------------------------------
def draw_text(img, text, x, y):
 cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
#-------------------------------------------------------------------------

def predict(test_img):

    img = test_img.copy()
    face, rect = detect_face(img)

  
    label= face_recognizer.predict(face)
    label_text = subjects[label[0]]
 

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
 
    return img



print("Predicting images...")

#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")
test_img4 = cv2.imread("test-data/test4.jpg")
test_img5 = cv2.imread("test-data/test5.jpg")
test_img6 = cv2.imread("test-data/test6.jpg")
#test_img7 = cv2.imread("test-data/test7.jpg")
#test_img8 = cv2.imread("test-data/test8.jpg")
#test_img9 = cv2.imread("test-data/test9.jpg")
#test_img10 = cv2.imread("test-data/test10.jpg")


#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)
predicted_img6 = predict(test_img6)
'''predicted_img7 = predict(test_img7)
predicted_img8 = predict(test_img8)
predicted_img9 = predict(test_img9)
predicted_img10 = predict(test_img10)'''

print("Prediction complete")


#display both images
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.imshow(subjects[3], predicted_img3)
cv2.imshow(subjects[4], predicted_img4)
cv2.imshow(subjects[5], predicted_img5)
cv2.imshow(subjects[6], predicted_img6)
'''cv2.imshow(subjects[7], predicted_img7)
cv2.imshow(subjects[8], predicted_img8)
cv2.imshow(subjects[9], predicted_img9)
cv2.imshow(subjects[10], predicted_img10)'''


cv2.waitKey(0)
cv2.destroyAllWindows()