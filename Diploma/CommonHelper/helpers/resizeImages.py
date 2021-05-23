import os
import cv2

root_dir = r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz'
width = 224
height = 224
dsize = (width, height)

print("Start")

ch_dirs = os.listdir(root_dir)
for i in ch_dirs:
    if i == 'Bad' or i == 'Good':
        p = os.path.join(root_dir, i)
        if (os.path.isdir(p)):
            filesDirs = os.listdir(p)
            for fileDir in filesDirs:
                    path = os.path.join(p, fileDir)
                    if (os.path.isdir(path)):
                        fileslist = os.listdir(path)
                        files = [os.path.join(path, f) for f in fileslist]
                        files = sorted(files, key=lambda x : os.path.getctime(x))
                        for file in files:
                            src = cv2.imread(file)
                            output = cv2.resize(src, dsize)
                            cv2.imwrite(file, output)
    else:
        p = os.path.join(root_dir, i)
        if (os.path.isdir(p)):
            files = os.listdir(p)
            files = [os.path.join(p, f) for f in files]
            files = sorted(files, key=lambda x : os.path.getctime(x))
            for file in files:
                src = cv2.imread(file)
                output = cv2.resize(src, dsize)
                cv2.imwrite(file, output)

print("End")

