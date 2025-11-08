import cv2
print(cv2.imread('data/data1.mp4') is None)   # will be True for videos (use VideoCapture)
cap = cv2.VideoCapture('data/data1.mp4'); print(cap.isOpened())
print(cv2.imread('data/secret.png') is None)