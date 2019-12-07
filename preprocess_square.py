import cv2
import numpy as np
import math
import os

show_Image = False
data_path = r"/home/pwrai/1124_train_photo"
target_path = "%s_preprocessed_square" % data_path
 

def panelAbstract(srcImage):
    #   read pic shape
    imgHeight, imgWidth = srcImage.shape[:2]
    imgHeight = int(imgHeight)
    imgWidth = int(imgWidth)
    # 二維轉一維
    imgVec = np.float32(srcImage.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, label, clusCenter = cv2.kmeans(imgVec, 2, None, criteria, 10, flags)
    clusCenter = np.uint8(clusCenter)
    clusResult = clusCenter[label.flatten()]
    imgres = clusResult.reshape(srcImage.shape)
    # image轉成灰階
    imgres = cv2.cvtColor(imgres, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("test.jpg", imgres)
    # image轉成2維，並做Threshold
    _, thresh = cv2.threshold(imgres, 127, 255, cv2.THRESH_BINARY_INV)

    threshRotate = cv2.merge([thresh, thresh, thresh])
    # 印出 threshold後的image
    # if cv2.imwrite(r"./Photo/thresh.jpg", threshRotate):
    #    print("Write Images Successfully")
    # 确定前景外接矩形
    # find contours
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minvalx = np.max([imgHeight, imgWidth])
    maxvalx = 0
    minvaly = np.max([imgHeight, imgWidth])
    maxvaly = 0
    maxconArea = 0
    maxAreaPos = -1
    for i in range(len(contours)):
        if maxconArea < cv2.contourArea(contours[i]):
            maxconArea = cv2.contourArea(contours[i])
            maxAreaPos = i

    print("Contours:", len(contours))

    if len(contours) > maxAreaPos:
        objCont = contours[maxAreaPos]
    else:
        print("Error: abnormal contours")
        return None  # return error code

    # cv2.minAreaRect生成最小外接矩形
    rect = cv2.minAreaRect(objCont)
    for j in range(len(objCont)):
        minvaly = np.min([minvaly, objCont[j][0][0]])
        maxvaly = np.max([maxvaly, objCont[j][0][0]])
        minvalx = np.min([minvalx, objCont[j][0][1]])
        maxvalx = np.max([maxvalx, objCont[j][0][1]])
    if rect[2] <= -45:
        rotAgl = 90 + rect[2]
    else:
        # 咖啡粉會執行else
        rotAgl = rect[2]
    if rotAgl == 0:
        panelImg = srcImage[minvalx:maxvalx, minvaly:maxvaly, :]
    else:
        # 咖啡粉會執行else

        _, dstRotBW = cv2.threshold(thresh, 127, 255, 0)
        # 印出最小外接矩形

        _, contours, hierarchy = cv2.findContours(dstRotBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0
        maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x, y, w, h = cv2.boundingRect(contours[maxAreaPos])
        # x，y是矩陣左上角的座標，w，h是矩陣的寬與高
        # umsize代表 1pixel*umsize = 真實大小
        umsize = 90000 / w
        #   print(umsize)
        w = w / 8  # 寬度分為8等分

        # 將沒有外圍輪廓的咖啡粉存入panelImg，固定照片大小，因此h以w代替
        panelImg = srcImage[int(y + 2 * w):int(y + 6 * w), int(x +2 * w):int(x + 6 * w), :]
        # 印出圖片真實大小
        print("Image Size:", 4 * w * umsize, " um * ", 4 * w * umsize, " um")
    return panelImg


def hist_equal_lab(img):
    global show_Image

    # Converting image. to LAB Color model
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)
    if show_Image:
        cv2.namedWindow('l_channel', cv2.WINDOW_NORMAL)
        cv2.imshow('l_channel', l)
        cv2.namedWindow("a_channel", cv2.WINDOW_NORMAL)
        cv2.imshow('a_channel', a)
        cv2.namedWindow("b_channel", cv2.WINDOW_NORMAL)
        cv2.imshow('b_channel', b)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    return cl


if __name__ == "__main__":

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    print("Target Path: ", target_path) 
    
    error_files = []
    

    for root_Outer, dirs_Outer, files_Outer in os.walk(data_path, topdown=False):
        for directory in dirs_Outer:
            target_dir = os.path.join(target_path, directory)

            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            print("Target Dir: ", target_dir)

            for root, dirs, files in os.walk(os.path.join(root_Outer, directory), topdown=False):
                for name in files:
                    img_path = os.path.join(root, name)
                    srcImage = cv2.imread(img_path)

                    (h, w) = srcImage.shape[:2]

                    center = (w/2, h/2)

                    for i in range(4):
                        rotation_matrix = cv2.getRotationMatrix2D(center, i*90, 1.0)
                        rstImage = cv2.warpAffine(srcImage, rotation_matrix, (w, h))

                        rstImage = panelAbstract(rstImage)

                        if rstImage is None:
                            error_files.append(img_path)
                            break

                        print(rstImage.shape)

                        rstImage = cv2.resize(rstImage, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                        print(rstImage.shape)

                        rstImage = hist_equal_lab(rstImage)
                        
                        # 印出結果
                        notcut_filename = '%s_result_%d' % (name.split('.')[0], i)
                        print('notcut_Filename: ' + notcut_filename)
                        print("Save in path: ", os.path.join(target_dir, notcut_filename))

                        for  i in range(4):
                            for j in range(4):

                                cutImage = rstImage[int(i*256):int((i+1)*256), int(j*256):int((j+1)*256)]
                                filename = notcut_filename + '[%d][%d].%s' %(i,j, name.split('.')[-1])
                                print('new_Filename: ' + filename)

                                if cv2.imwrite(os.path.join(target_dir, filename), cutImage):
                                    print("Write Images Successfully") 
                        
                    print("Error Files:", error_files)
