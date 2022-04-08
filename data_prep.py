import os
import shutil
import random

random.seed(5672349)

spec_dir = "GTZAN/images_original"


def make_or_clean_dir(d):
    try:
        os.mkdir(d)
    except FileExistsError:
        shutil.rmtree(d)
        os.mkdir(d)


data_folders = ["data/train", "data/test"]
make_or_clean_dir("data")
for d in data_folders:
    make_or_clean_dir(d)

for genre in os.listdir(spec_dir):
    spec_names = list(os.listdir(os.path.join(spec_dir, genre)))
    random.shuffle(spec_names)
    test_paths = spec_names[:20]
    train_paths = spec_names[20:]

    for d in data_folders:
        make_or_clean_dir(os.path.join(d, genre))

    for spec_name in test_paths:
        shutil.copyfile(
            os.path.join(spec_dir, genre, spec_name),
            os.path.join(data_folders[1], genre, spec_name),
        )

    for spec_name in train_paths:
        shutil.copyfile(
            os.path.join(spec_dir, genre, spec_name),
            os.path.join(data_folders[0], genre, spec_name),
        )

# done.
