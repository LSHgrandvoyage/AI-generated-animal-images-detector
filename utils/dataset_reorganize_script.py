import os
import shutil
import random
from dotenv import load_dotenv

def ai_unclassify(path):
    for split in ['train', 'val', 'test']:
        target_path = os.path.join(path, split, 'ai', 'sd_turbo')
        counter = 1
        for class_ in os.listdir(target_path):
            class_dir = os.path.join(target_path, class_)
            for filename in os.listdir(class_dir):
                src = os.path.join(class_dir, filename)
                dst_tag = f"ai_image_{counter:04d}.png"
                dst = os.path.join(target_path, dst_tag)

                shutil.move(src, dst)
                counter += 1
            os.rmdir(class_dir)
        print(f"{split} Dataset(AI) unclassified.")

def real_unclassify(path, dataset_path):
    split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    all_images = []

    for class_ in os.listdir(path):
        class_dir = os.path.join(path, class_)
        for filename in os.listdir(class_dir):
            all_images.append(os.path.join(class_dir, filename))

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * split_ratio['train'])
    val_end = int(total * (split_ratio['train'] + split_ratio['val']))

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    counter = 1
    for split, files in splits.items():
        split_dir = os.path.join(dataset_path, split, 'real')
        for src in files:
            dst_tag = f"real_image_{counter:04d}.png"
            dst = os.path.join(split_dir, dst_tag)
            shutil.move(src, dst)
            counter += 1
        print(f"{split} Dataset(REAL) unclassified.")

def final_rename(path):
    for split in ['train', 'val', 'test']:
        for data_type in ['real', 'ai']:
            target_path = os.path.join(path, split, data_type)
            if data_type == 'ai':
                target_path = os.path.join(path, split, data_type, 'sd_turbo')

            files = [f for f in os.listdir(target_path) if f.lower().endswith('.png')]
            random.shuffle(files)
            for i, filename in enumerate(files, start=1):
                old_path = os.path.join(target_path, filename)
                temp_name = f"temp_{i:04d}.png"
                temp_path = os.path.join(target_path, temp_name)
                os.rename(old_path, temp_path)

            temp_files = [f for f in os.listdir(target_path) if f.lower().endswith('.png')]
            prefix = f"{data_type}_image_"
            for i, filename in enumerate(temp_files, start=1):
                old_path = os.path.join(target_path, filename)
                new_tag = f"{prefix}{i:04d}.png"
                new_path = os.path.join(target_path, new_tag)
                os.rename(old_path, new_path)
        print(f"{split} Dataset renamed.")

if __name__ == '__main__':
    load_dotenv()
    real_data_dir = os.getenv('TEMP_PATH')
    data_dir = os.getenv('DATA_PATH')
    ai_unclassify(data_dir)
    real_unclassify(real_data_dir, data_dir)
    final_rename(data_dir)
