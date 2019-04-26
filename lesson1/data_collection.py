import cv2
import os

capture = cv2.VideoCapture(0) #-> 0->1 for usb camera
dataset_path = "./dataset/"
if not (os.path.exists(dataset_path)):
    os.mkdir(dataset_path)

for i in range(3):
    class_path = dataset_path + str(i) 
    if not (os.path.exists(class_path)):
        os.mkdir(class_path)

data_counter = 0
class_number = 0
while(True):
    ret, frame = capture.read()
    show_img = frame.copy()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(show_img, (100,100), (200, 200), (0, 0, 255), 5)
    crop_img = frame[100:200, 100:200]
    cv2.imshow('frame', show_img)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        file_name = dataset_path + str(class_number) + "/" + "%03d.jpg" %data_counter
        cv2.imwrite(file_name, crop_img)
        print "saving picture %d" %data_counter
        data_counter = data_counter + 1

    elif k == ord('n'):
        class_number = class_number + 1
        data_counter = 0
        print "saving class %d" %class_number

capture.release()
cv2.destroyAllWindows()
