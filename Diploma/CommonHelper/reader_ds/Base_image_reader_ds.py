
from reader_ds.Base_reader_ds import Base_reader_ds

class Base_image_reader_ds(Base_reader_ds):
    def __init__(self, 
                 augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds,
                 type_load_im, shape_to_resize, num_chanels_input, coef_normalize
                 ):
        super().__init__(augmentation, path_to_mapfile, percent_slice, step_folder, desired_size_ds)
        self._type_load_im = type_load_im
        self._shape_to_resize = shape_to_resize 
        self._num_chanels_input = num_chanels_input 
        self._coef_normalize = coef_normalize