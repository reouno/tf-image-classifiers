import glob
import os
from typing import Text

def list_dir(dirpath: Text):
    '''cout no. of directories under the `dirpath`
    :param dirpath: directory path
    '''

    dirs = [d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,d))]
    return dirs

def get_all_files(dir_path: Text):
    '''get all file paths under the directory recursively
    '''
    files = glob.glob(os.path.join(dir_path, '**'), recursive=True)
    files = [fp for fp in files if os.path.isfile(fp)]
    return files

def get_all_img_files(dir_path: Text):
    '''get all image file paths under the directory recursively
    '''
    img_formats = set(['.PNG','.JPG','.JPEG','.BMP','.PPM','.TIF'])
    files = glob.glob(os.path.join(dir_path, '**'), recursive=True)
    img_files = [fp for fp in files if os.path.splitext(fp)[1].upper() in img_formats]
    return img_files

