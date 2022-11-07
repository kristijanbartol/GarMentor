import argparse
import os
from os import path as osp
from tqdm import tqdm

def fix_materials(base_dir: str) -> None:
    """Applies the material fix to all obj files inside of the provided
    directory and all of its subdirectories.
    """
    if not osp.isdir(base_dir):
        raise ValueError("The provided path is not a directory!")
    for element in tqdm(os.listdir(base_dir), desc=f"Processing {base_dir}"):
        if osp.isdir(osp.join(base_dir, element)):
            fix_materials(osp.join(base_dir, element))
        elif osp.isfile(osp.join(base_dir, element)):
            if osp.splitext(element)[1] == '.obj':
                _apply_fix(osp.join(base_dir, element))


def _apply_fix(obj_fpath):
    obj_fname = osp.splitext(osp.basename(obj_fpath))[0]
    content = ""
    with open(obj_fpath, "r") as obj_file:
        first_face = True
        usemtl_encountered = False
        for line in obj_file:
            line_splitted = line.strip().split(' ')
            if not usemtl_encountered and line_splitted[0] == 'usemtl':
                usemtl_encountered = True
            if first_face and line_splitted[0] == 'f':
                first_face = False
                if not usemtl_encountered:
                    content += f"usemtl {obj_fname}\n"
                    usemtl_encountered = True
            content += line
    with open(obj_fpath, "w") as obj_file:
        obj_file.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fixes the material application for all obj files"
    )
    parser.add_argument('-d', '--directory', type=str, help="Apply the "
    "material fix to all obj files in this directory and subdirectories.")
    args = parser.parse_args()
    fix_materials(args.directory)
