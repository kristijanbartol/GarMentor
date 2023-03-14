from typing import List
import tqdm
import os


def get_background_paths(
        backgrounds_dir_path: str, 
        num_backgrounds: int = -1
    ) -> List[str]:
    print('Loading background paths...')
    backgrounds_paths = []
    for f in tqdm(sorted(os.listdir(backgrounds_dir_path)[:num_backgrounds])):
        if f.endswith('.jpg'):
            backgrounds_paths.append(
                os.path.join(backgrounds_dir_path, f)
            )
    return backgrounds_paths
