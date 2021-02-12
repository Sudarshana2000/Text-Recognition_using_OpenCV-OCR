import cv2
import numpy as np
import pytesseract
import argparse

pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
east="model/frozen_east_text_detection.pb"
config = ("-l eng --oem 1 --psm 7")

def four_point_transform(image,points):
    (bl,tl,tr,br)=points
    rect=np.asarray([tl,tr,br,bl])
    width1=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    width2=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    width=max(int(width1),int(width2))
    height1=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    height2=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    height=max(int(height1),int(height2))
    dest=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(rect,dest)
    warped=cv2.warpPerspective(image,M,(width,height))
    return warped    

def decodeBoundingBoxes(scores, geometry, scoreThresh,padding):
    detections = []
    confidences = []
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]
        for x in range(0, width):
            score = scoresData[x]
            if (score < scoreThresh):
                continue
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            p3 = (offsetX + cos * xData1[x] + sin * xData2[x], offsetY - sin * xData1[x] + cos * xData2[x])            
            p1 = (- sin*h - cos*w + p3[0], p3[1] + w*sin - h*cos)
            #p2 = (-sin * h + p3[0], -cos * h + p3[1])
            #p4 = (-cos * w + p3[0], sin * w + p3[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w*(1+padding), h*(1+padding)), -1 * angle * 180.0 / np.pi))
            confidences.append(float(score))
    return [detections, confidences]

def text_detector(image,new_size=(320,320),conf_thres=0.5,nms_thres=0.4,padding=0):
    img=image.copy()
    h,w=img.shape[:2]
    (H,W)=new_size
    rH,rW=h/H,w/W
    img=cv2.resize(img,(W,H))
    detector = cv2.dnn.readNet(east)
    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    detector.setInput(blob)
    layerNames=["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
    (scores, geometry) = detector.forward(layerNames)
    [boxes, confidences] = decodeBoundingBoxes(scores, geometry,conf_thres,padding)
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, conf_thres, nms_thres)
    img1=image.copy()
    results=[]
    for i in indices:
        vertices=cv2.boxPoints(boxes[i[0]])
        for j in range(4):
            vertices[j][0]=int(vertices[j][0]*rW)
            vertices[j][1]=int(vertices[j][1]*rH)
        for j in range(4):
            res=cv2.line(img1,tuple(vertices[j]),tuple(vertices[(j+1)%4]),(0,255,0),2)
        roi=four_point_transform(image.copy(),vertices)
        roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi=cv2.medianBlur(roi,5)
        text = pytesseract.image_to_string(roi, config=config)
        results.append((vertices, text))
    #cv2.imshow("output1",img1)
    #cv2.waitKey(0)
    results=sorted(results, key=lambda r:r[0][1][1])
    for points,value in results:
        text = "".join([c if ord(c) < 128 else "" for c in value]).strip()
        print(text,end=" ")
        if points[1,1]-20>0:
            res=cv2.putText(img1, text, (int(points[1,0]), int(points[1,1]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            res=cv2.putText(img1, text, (int(points[0,0]), int(points[0,1]) + 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    #cv2.imshow("output2",img1)
    #cv2.waitKey(0)
    return img1

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", required=True, type=str, help="path to input image")
    ap.add_argument("output_path", required=True, type=str, help="path to output image")
    ap.add_argument("tesseract_path", type=str, help="path to tesseract OCR")
    ap.add_argument("east", type=str, help="path to east-text-detection model")
    ap.add_argument("confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
    ap.add_argument("padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
    args = vars(ap.parse_args())
    pytesseract.pytesseract.tesseract_cmd=args.tesseract_path
    global east
    east = args.east
    img = cv2.imread(args.image_path)
    output = text_detector(img,args.confidence,0.4,args.padding)
    cv2.imwrite(args.output_path, output)