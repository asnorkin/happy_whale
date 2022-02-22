import os
import os.path as osp


def create_if_not_exist(dirpath: str):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)
