import os
import glob
import cv2


def getCroped(oriImgPath, croppedImgName, targetPath, x1, y1, x2, y2):
    # Read image
    img = cv2.imread(oriImgPath)
    # get the height and width of the image
    (h, w) = img.shape[:2]
    # get the coordinates from x--y-w-h
    ix = int(x1 * w)
    iy = int(y1 * h)
    iw = int(x2 * w / 2)
    ih = int(y2 * h / 2)
    x2 = ix + iw
    x1 = ix - iw
    y2 = iy + ih
    y1 = iy - ih
    # Crop image
    crop_img = img[y1:y2, x1:x2]
    # Save image
    print([y1,y2, x1,x2])
    cv2.imwrite(targetPath + "/" + croppedImgName + ".jpg", crop_img)


def cropAll(folderPath, targetPath):
    # find all files under this path in jpg format
    images = glob.glob(folderPath + "/*.jpg")
    for oriImg in images:
        # get the name of the image
        imgName = os.path.basename(oriImg)
        # get the name of the txt file
        txtName = os.path.splitext(imgName)[0] + ".txt"
        # get the path of the txt file
        txtPath = folderPath + "/labels/" + txtName
        # open the txt file
        f = open(txtPath, "r")
        # read the txt file
        lines = f.readlines()
        # get the first line
        line = lines[0]
        # split the line
        line = line.split(' ')
        # get the coordinates
        x1 = float(line[1])
        y1 = float(line[2])
        x2 = float(line[3])
        y2 = float(line[4])
        # close the txt file
        f.close()
        # crop the image
        getCroped(oriImg, os.path.splitext(imgName)[0], targetPath, x1, y1, x2, y2)

cropAll("./runs/detect/exp", "./runs2")