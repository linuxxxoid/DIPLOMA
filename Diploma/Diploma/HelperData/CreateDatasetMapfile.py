import os


### Softmax
save_dir = r'D:\mine\diploma\Dataset\Softmax'
name_map_file = 'map_file.txt'
path_map_file = os.path.join(save_dir, name_map_file)
bad_paths = [r'D:\mine\diploma\Dataset\Data\Skeleton\Other']

good_paths = [r'D:\mine\diploma\Dataset\Data\Skeleton\Vertic\Good',
			  r'D:\mine\diploma\Dataset\Data\Skeleton\Vertic\Bad',
			  r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Good',
			  r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Bad']

### Vertic triplet
#save_dir = r'D:\mine\diploma\Dataset\Triplet\Vertic_triplet'
#name_map_file = 'map_file.txt'
#path_map_file = os.path.join(save_dir, name_map_file)
#bad_paths = [r'D:\mine\diploma\Dataset\Data\Skeleton\Vertic\Bad',
#             r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Good']

#good_paths = [r'D:\mine\diploma\Dataset\Data\Skeleton\Vertic\Good',
#              ]

###Horiz triplet
#save_dir = r'D:\mine\diploma\Dataset\Triplet\Horiz_triplet'
#name_map_file = 'map_file.txt'
#path_map_file = os.path.join(save_dir, name_map_file)
#bad_paths = [r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Bad',
#             r'D:\mine\diploma\Dataset\Data\Skeleton\Vertic\Good']

#good_paths = [r'D:\mine\diploma\Dataset\Data\Skeleton\Horiz\Good',
#              ]

index_class = 1

print("Start")
with open(path_map_file, 'w') as f:
    for gp in good_paths:
        good_files = os.listdir(gp)
        for gf in good_files:
            p = os.path.join(gp, gf)
            if (os.path.isdir(p) and gf.lower() == 'bad' or gf.lower() == 'good'):
                filesDirs = os.listdir(p)
                for fileDir in filesDirs:
                    path = os.path.join(p, fileDir)
                    f.write(path + '\t' + str(index_class) + '\n')
            else:
                f.write(p + '\t' + str(index_class) + '\n')
    index_class -= 1
    for bp in bad_paths:
        bad_files = os.listdir(bp)
        for bf in bad_files:
            p = os.path.join(bp, bf)
            if (os.path.isdir(p) and bf.lower() == 'bad' or bf.lower() == 'good'):
                filesDirs = os.listdir(p)
                for fileDir in filesDirs:
                    path = os.path.join(p, fileDir)
                    f.write(path + '\t' + str(index_class) + '\n')
            else:
                f.write(p + '\t' + str(index_class) + '\n')
print("End")

