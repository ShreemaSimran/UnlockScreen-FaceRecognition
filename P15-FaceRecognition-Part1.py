import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):           #from video extract the face
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          #convert to grayscale
    faces = face_classifier.detectMultiScale(gray,1.3,5)           #taken neighbours as 5 since frame is smalll

    if faces is():
        return None                    #in case there is no face in frame

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]           #create the rectangle or frame formy face(defining the length without marking rectangle)

    return cropped_face


cap = cv2.VideoCapture(0)                   #opens web cam
count = 0                                   #count the images

while True:
    ret, frame = cap.read()                 #read the frame
    if face_extractor(frame) is not None:            #if there is face in frame then do
        count+=1                                              #count the images
        face = cv2.resize(face_extractor(frame),(200,200))              #resize the image to particular frame(optional)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)                   #display image in gray

        file_name_path = 'E:/Data/user'+str(count)+'.jpg'              #location to store .jpg captured image
        cv2.imwrite(file_name_path,face)                                #creating the image file

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)        #display number on the screen
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")           #display message if no face
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # until we press enter or count should be 100
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete!!!')