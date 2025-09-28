import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
pd.set_option('display.max_rows', None)

csv_file_path = os.getenv("CSV_FILE_PATH")
class_path = os.getenv("CLASS_PATH")

def load_dataframe():
    df = pd.read_csv(csv_file_path)
    return df

def add_species_column(df, path):
    """
    Function to add [species] column to dataframe
    """
    with open(path, 'r') as f:
        class_map = json.load(f)
    df['class'] = df['class'].astype(str)
    df['species'] = df['class'].map(class_map)
    return df

def show_image_species(df):
    """
    Function to display image counts by species
    """
    species_counts = df['species'].value_counts()
    print(species_counts)

def count_images_by_generator(df):
    """
    Function to display image counts by generator
    """
    t_sum = 0
    t_count = 0

    generator_counts = df['generator'].value_counts()
    for generator, count in generator_counts.items():
        if generator == "nature":
            t_count = count
            t_sum += count
            continue
        print(f"{generator:<25} {count:,}")

    print(f"The number of nature images: {t_count:,}")
    print(f"The number of fake images: {t_sum:,}")

if __name__ == '__main__':
    df = load_dataframe()
    count_images_by_generator(df)
    #df = add_species_column(df, class_path)
    #show_image_species(df)


