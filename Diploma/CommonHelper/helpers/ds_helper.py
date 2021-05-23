import os



# создать мапфайл для ds классификации
def create_mapfile_class_ds(path_to_root_dir, is_sorted_by_index = True):
    exts = ('png', 'jpg', 'tiff')
    name_map_file = 'map_file.txt'
    path_map_file = os.path.join(path_to_root_dir, name_map_file)
    index_class = 0

    with open(path_map_file, 'w') as f:
        ch_dirs = os.listdir(path_to_root_dir)
        for i in ch_dirs:
            p = os.path.join(path_to_root_dir, i)
            if (not os.path.isdir(p)):
                continue
            files_ims = [f for f in os.listdir(p) if f.endswith(exts)]
            if is_sorted_by_index:
                files_ims = sorted(files_ims, key=lambda x : int(('.').join(x.split('.')[:-1])) )
            files_ims = [os.path.join(p, f) for f in files_ims]
            for file in files_ims:
                f.write(file + '\t' + str(index_class) + '\n')
            index_class += 1
            


if __name__ == '__main__':

    create_mapfile_class_ds(r'')

    print('Done')