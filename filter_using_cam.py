import cv2
import numpy as np


cap=cv2.VideoCapture(0)


img=cv2.imread("E:\\programs code\\env2\\Project\\witch2.png")
img=cv2.resize(img,(200,100))



r,c,h=img.shape

classifier1 = cv2.CascadeClassifier("E:\programs code\env2\Project\\haarcascade.xml")

img_g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,mask= cv2.threshold(img_g,10,255,cv2.THRESH_BINARY) # if thr value is less than 10=> 0, otherwise 255
mask_i=cv2.bitwise_not(mask)
cv2.imshow("mask_i",mask_i)
cv2.imshow("mask",mask)

while True:
  ret, frame=cap.read()
  
  if ret:
    
    faces = classifier1.detectMultiScale(frame)
    for face in faces:
     x, y, w, h = face
     # img_w=int(1.5*(x+w))
     # img_h=int(1.5*(y+h))
     frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
    
     # img=cv2.resize(img,(x+w,y+h))
    
     
     # w2=3*w
     # h2=int(1.5*h)
    
     # hx1=x
     # hx2=hx1+w2
     # hy1=y
     # hy2=hy1+h2
     
     w2=w
     h2=h
    
     hx1=x
     hx2=hx1+w2
     hy1=y
     hy2=hy1+h2

     # roi=frame[:20+r,20:20+c]
     
     img = cv2.resize(img, (hx2-hx1,hy2-hy1), interpolation = cv2.INTER_AREA)
     mask = cv2.resize(mask, (hx2-hx1,hy2-hy1), interpolation = cv2.INTER_AREA)
     mask_i = cv2.resize(mask_i, (hx2-hx1,hy2-hy1), interpolation = cv2.INTER_AREA)
     if(hy1-h<0):
      continue
     roi=frame[(hy1+10)-(h+10):hy1,hx1:hx2]
     frame_bg=cv2.bitwise_and(roi,roi,mask=mask_i)   # bitwise operation in only that region where mask is present  (non zero)and in reference to that  
     frame_fg=cv2.bitwise_and(img,img,mask=mask)
     fr,fc,fch=frame_fg.shape
     dst=frame_bg
   
     
     frame[hy1-h:hy1,hx1:hx2]=dst
     cv2.imshow("bg", frame_bg)
     cv2.imshow("fg", frame_fg) 
     cv2.imshow("My window", dst)
  
    key=cv2.waitKey(2)
    if key==ord('q'):
      break;                                                                                            
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows() 