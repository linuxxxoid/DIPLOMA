
import os

exts = ['png', 'jpg', 'tiff']
root_dir = (r'D:\datasets\frog_sequences')
name_map_file = 'map_file.txt'
path_map_file = os.path.join(root_dir, name_map_file)
index_class = 0

print("Start")
with open(path_map_file, 'w') as f:
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
                                f.write(file + '\t' + str(index_class) + '\n')
                index_class += 1
        else:
            p = os.path.join(root_dir, i)
            if (os.path.isdir(p)):
                files = os.listdir(p)
                files = [os.path.join(p, f) for f in files]
                files = sorted(files, key=lambda x : os.path.getctime(x))
                for file in files:
                    f.write(file + '\t' + str(index_class) + '\n')
                index_class += 1
print("End")
