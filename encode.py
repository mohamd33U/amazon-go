import os
import cv2 as cv
import face_recognition as fr
import pickle

folderimgpath = "faces"
pathlist = os.listdir(folderimgpath)
imglist = []
clineIds = []

# Load images and corresponding IDs
for path in pathlist:
    imglist.append(cv.imread(os.path.join(folderimgpath, path)))
    clineIds.append(os.path.splitext(path)[0])

# Function to find face encodings
def findencoding(imglist):
    encodelist = []
    for img in imglist:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB
        encode = fr.face_encodings(img_rgb)[0]
        encodelist.append(encode)
    return encodelist

encodelistknew = findencoding(imglist)
knewandid = [encodelistknew, clineIds]

# Save both encodelistknew and clineIds
file = open("encodefile.p", "wb")
pickle.dump(knewandid, file)
file.close()
