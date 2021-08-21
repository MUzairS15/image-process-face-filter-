import cv2
import numpy as np
frame=cv2.imread("testimg.png")
frame=cv2.resize(frame,(960,420))
print(frame.shape)
img=cv2.imread("imgh.png")
img=cv2.resize(img,(200,100))

# img[:,:,:]=frame.shape
r,c,ch=img.shape

img_g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_w=int(0.5*c)
img_h=int(0.5*r)
roi=frame[30:30+r,30:30+c]
ret,mask= cv2.threshold(img_g,10,255,cv2.THRESH_BINARY)
mask_i=cv2.bitwise_not(mask)
frame_bg=cv2.bitwise_and(roi,roi,mask=mask_i)     
frame_fg=cv2.bitwise_and(img,img,mask=mask)
     
dst=cv2.add(frame_bg,frame_fg)
print("dst",dst.shape)
print("img",img.shape)
frame[50:50+r,80:80+c]=dst
   
cv2.imshow("My window", frame)

cv2.waitKey(0);

cv2.destroyAllWindows()