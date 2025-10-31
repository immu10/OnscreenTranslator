import cv2
import pytesseract
# import imutils
import mss
import numpy as np
 
# image = cv2.imread("image.png")
 
# image_og = imutils.resize(image,width=300)
# image_og = image
# cv2.namedWindow("input")
# cv2.namedWindow("grey")
 
 
# cv2.imshow("input", image)
# cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord("q"):
#     break
 
# image = image_og.copy()
# gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# grey_img = cv2.bilateralFilter(gray_img,11,17,17)
# edge = cv2.Canny(gray_img,30,200)  # black and white
 
# cnts,new = cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# image_blu = image_og.copy()
# cv2.drawContours(image_blu,cnts,-1,(255,0,0),3)
 
# image2 = image_og.copy()
# cnts = sorted(cnts,key = cv2.contourArea,reverse=True)  # to select top contours 30 random number
# cv2.drawContours(image2,cnts,-1,(255,0,0),3)
 
 
# image3 = image_og.copy()
 
# i = 7
# screencnt = None
# approx_list = []
# for c in cnts:
#     perimter = cv2.arcLength(c,True)
#     approx = cv2.approxPolyDP(c,0.02 * perimter, True )
#     print(len(approx))
#     approx_list.append(approx)
#     if len(approx) == 4:
#         print("found one")
#         screencnt = approx
#         x,y,w,h = cv2.boundingRect(c)
#         new_img = image_og[y:y+h,x:x+w]
#         # cv2.imwrite('./'+str(i)+".png",new_img)
#         i+= 1
        # continue
 
# i = 1
# for approx in approx_list:
#     # print(approx_list)
#     x,y,w,h = cv2.boundingRect(approx)
#     windowName=f"{i} item"
#     cv2.namedWindow(windowName)
#     pic = cv2.rectangle(image_og.copy(),(x,y),(x+w,y+h), (255,0,0), 2 )
#     cv2.imshow(windowName, pic)
#     cv2.waitKey(0)
#     pic = None
 
# cv2.drawContours(image_og,[screencnt],-1,(0,255,0),3)
 
 
# cv2.namedWindow("grey")
# cv2.imshow("grey", edge)
# cv2.waitKey(0)
import datetime

import easyocr
reader = easyocr.Reader(['en'], gpu=True)
 
 
sct = mss.mss()
monitor = sct.monitors[1]
 
custom_tessdata = "tessdata"
custom_lang = "eng"
cntr = 1
scale = 0.5
while True:
    sct_frame = sct.grab(monitor)
    if cntr < 20:
        cntr+=1
        continue
    img = np.array(sct_frame)
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
    imgH,imgW,_ = img.shape
    new_width = int(imgW * scale)
    new_height = int(imgH * scale)
    img = cv2.resize(img, (new_width, new_height))
 
    # print("starting at", datetime.datetime.now())
    # imgboxes = pytesseract.image_to_boxes(img,lang= custom_lang, config= f"--tessdata-dir {custom_tessdata}")
    imgboxes = reader.readtext(img,paragraph=True)
    # print("ending at", datetime.datetime.now())
   
    for word in imgboxes:
        point1,_,point2,_ = word[0]
        point1,point2 = tuple(point1),tuple(point2)
        cv2.rectangle(img, pt1=point1 , pt2=point2 , color=(255,0,0), thickness=3)
 
 
    """ tesseract vvv"""
    # for boxes in imgboxes.splitlines():
    #     boxes = boxes.split( ' ')
    #     x,y,w,h = int(boxes[1]), int(boxes[2]),int(boxes[3]),int(boxes[4])
    #     cv2.rectangle(img,(x,imgH-y),(w,imgH-h),(255,0,0),3)
    #     # cv2.imshow("frame",img)
    #     # print(boxes)
    #     # if cv2.waitKey(2) & 0xFF == ord('q'):
    #     #     break
    #     # print(boxes)
   
   
   
    cv2.imshow("frame",img)
 
    if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    img = None
    cntr = 1
 
# print(text)
 
# import cv2
# import easyocr
# import matplotlib.pyplot as plt
# import numpy as np
# import datetime
 
 
# img = cv2.imread("image.png")
# # img = image
 
# # instance text detector
# reader = easyocr.Reader(['en'], gpu=True)
 
# # detect text on image
# print("starting now", datetime.datetime.now())
# text_ = reader.readtext(img)
# print("read now", datetime.datetime.now())
 
# threshold = 0.25
# # draw bbox and text
# for t_, t in enumerate(text_):
#     print(t)
 
#     bbox, text, score = t
 
#     if score > threshold:
#         print("normalizing now", datetime.datetime.now())
#         pt1 = tuple(map(int, bbox[0]))
#         pt2 = tuple(map(int, bbox[2]))
#         print("normalized now", datetime.datetime.now())
#         cv2.rectangle(img, pt1, pt2, (0, 255, 0), 5)
#         cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
 
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()