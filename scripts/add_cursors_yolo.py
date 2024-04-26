import os
import shutil
import random
import glob
from PIL import CurImagePlugin, Image
import click
from torch.utils.data import Dataset, Subset
import pandas as pd
import numpy as np
import yaml

def place_cursor(cur, screen, position):
    merged = Image.new("RGBA", screen.size)
    mask  = cur.split()[-1]
    merged.paste(screen, (0, 0), None)
    merged.paste(cur, position, mask)
    return merged, (position[0], position[1], position[0] + cur.size[0], position[1] + cur.size[1])

def get_file_names(path):
    return [name.split('.')[0] for name in glob.glob(os.path.join(path, '*'))]

def generate_cursor_bbox(screen_img, bboxes, cursor_set):
    # This code generates a random bbox and returns a cursor
    all_cursors = cursor_set.get_all_cursors()
    cur_idx = random.randint(0,len(all_cursors)-1) # choose random cursor
    cur_img = all_cursors[cur_idx][1]

    w = cur_img.size[0]/screen_img.size[0]
    h = cur_img.size[1]/screen_img.size[1]
    x = random.uniform(0, 1-w)
    y = random.uniform(0, 1-h)

    return (x, y, w, h), cur_img

class CursorSet:
    def __init__(self, path):
        self.path = path
        self.all_cursors = []
        for cur_path in glob.glob(os.path.join(path, '*.[jp][pn]g')):
            cur_img = Image.open(cur_path)
            cursor = (os.path.splitext(os.path.basename(cur_path))[0], cur_img)
            self.all_cursors.append(cursor)
    
    def get_beam_cursors(self,):
        raise NotImplementedError()

    def get_non_hover_cursors(self,):
        raise NotImplementedError()
    
    def get_loading_cursors(self,):
        raise NotImplementedError()

    def get_all_cursors(self,):
        return self.all_cursors

class ScreenDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.screen_names, self.img_exts = self._get_screens()

    def _get_screens(self):
        imgs_path = os.path.join(self.path, 'images')
        labels_path = os.path.join(self.path, 'labels')
        imgs = glob.glob(os.path.join(imgs_path, '*'))
        labels = glob.glob(os.path.join(labels_path, '*'))
        imgs = [os.path.splitext(os.path.basename(p)) for p in imgs]
        img_exts = dict(imgs) # dict of name : extension
        labels = [os.path.splitext(os.path.basename(p))[0] for p in labels]
        return list(set(img_exts).intersection(set(labels))), img_exts

    def __getitem__(self, idx):
        name = self.screen_names[idx]
        img_path = os.path.join(self.path, 'images', name + self.img_exts[name])
        screen = Image.open(img_path)
        label_path = os.path.join(self.path, 'labels', name + ".txt")
        bboxes = pd.read_csv(label_path, sep=' ', names=['class','x','y','w','h'],
        dtype={
            'class': np.int32,
            'x': np.float64,
            'y': np.float64,
            'w': np.float64,
            'h': np.float64,
        })
        return screen, bboxes, img_path, label_path

    def __len__(self):
        return(len(self.screen_names))
    
    def choices(self, indicies):
        new_names = set()
        name_counter = dict()
        new_exts = dict()
        for i in indicies:
            name = self.screen_names[i]
            if name in new_names:
                aug_name = name + f"_{name_counter[name]}"
                name_counter[name] += 1
                new_exts[aug_name] = self.img_exts[name]
            else:
                new_names.add(name)
                name_counter[name] = 1
                new_exts[aug_name] = self.img_exts[name]

        self.screen_names = new_names
        self.img_exts = new_exts


@click.command()
@click.argument('cursors', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('data_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('dst', type=click.Path(exists=True, writable=True, file_okay=False, dir_okay=True))
@click.option('--num_train', type=int, default=100)
@click.option('--num_val', type=int, default=50)
@click.option('--num_test', type=int, default=50)
@click.option('--repeat', default=False, is_flag=True)
@click.option('--cursors_only', default=False, is_flag=True)
@click.option('--rand_seed', type=int, default=1234)
def script(cursors, data_path, dst, num_train, num_val, num_test, repeat, cursors_only, rand_seed):
    random.seed(rand_seed)
    
    with open(data_path, 'r') as file:
        data_config = yaml.safe_load(file)
    
    if 'cursor' not in data_config['names']:
        data_config['names'].append('cursor')
        data_config['nc'] += 1

    # TODO Yaml file has the train, test, validation section maybe remove or something?
    new_class=data_config['names'].index('cursor')

    cursor_set = CursorSet(cursors)

    num = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }

    for section in ['train', 'val', 'test']:
        section_path = data_config.get(section, None)
        if not section_path:
            continue
        if not os.path.isabs(section_path):
            section_path = os.path.join(os.path.dirname(data_path), section_path)
        if not os.path.exists(section_path):
            raise ValueError(f"Data configuration file given in {data_path} does not contain an existing {section} path {section_path}")
        data = ScreenDataset(section_path)

        if repeat:
            subset_indicies = sorted(random.choices(range(len(data)), num[section]))
        else:
            subset_indicies = sorted(random.sample(range(len(data)), num[section]))

        data = Subset(data, indices=subset_indicies)

        section_dst = os.path.join(dst, section)
        os.makedirs(section_dst, exist_ok=True)

        images_path = os.path.join(section_dst, 'images')
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        labels_path = os.path.join(section_dst, 'labels')
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)

        for screen, bboxes, img_path, label_path in data:

            random_bbox, cur_img = generate_cursor_bbox(screen, bboxes, cursor_set)

            x, y, w, h = random_bbox

            position = (int(x*screen.size[0]), int(y*screen.size[1]))

            new_screen, _ = place_cursor(cur_img, screen, position)

            # Yolo uses mid points as positions
            if cursors_only:
                cursor_bbox_df = pd.DataFrame({"class": new_class, "x": x+w/2, "y": y+h/2, 'w': w, 'h': h}, index=[0])
                cursor_bbox_df.to_csv(os.path.join(section_dst, 'labels', os.path.basename(label_path)), sep=' ', index=False, header=False)
            else:
                bboxes.loc[len(bboxes)] = {"class": new_class, "x": x+w/2, "y": y+h/2, 'w': w, 'h': h}
                bboxes.to_csv(os.path.join(section_dst, 'labels', os.path.basename(label_path)), sep=' ', index=False, header=False)


            img_ext = os.path.splitext(img_path)[1]

            if img_ext == '.jpg':
                new_screen.convert('RGB').save(os.path.join(section_dst, 'images', os.path.basename(img_path)))
            else:
                new_screen.save(os.path.join(section_dst, 'images', os.path.basename(img_path)))
            data_config[section] = section_dst

    with open(os.path.join(dst, 'gravity.yaml'), 'w') as file:
        yaml.dump(data_config, file)


if __name__ == "__main__":
    script()