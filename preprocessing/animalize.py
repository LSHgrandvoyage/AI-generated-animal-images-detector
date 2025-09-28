import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from nltk.corpus import wordnet as wn
from show_data import *

def is_animal(word):
    """
    Function to check whether a word is an animal
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    for syn in synsets:
        if any('animal' in h.name() or 'living_thing' in h.name() for h in syn.closure(lambda s: s.hypernyms())):
            return True
    return False

def count_animal_images(df, path):
    """
    Function to count the number of animal images
    """
    df = add_species_column(df, path)

    df['is_animal'] = df['species'].apply(lambda x: is_animal(x) if isinstance(x, str) else False)
    animal_df = df[df['is_animal']]

    real_animal_df = animal_df[animal_df['generator'] == 'nature']
    fake_animal_df = animal_df[animal_df['generator'] != 'nature']

    print(f"The number of animal images : {len(animal_df):,}")
    print(f"The number of real animal images : {len(real_animal_df):,}")
    print(f"The number of fake animal images : {len(fake_animal_df):,}")

    save_species_counts(real_animal_df, fake_animal_df)
    #visualize_species_distribution(real_animal_df, fake_animal_df)

def save_species_counts(real_df, fake_df):
    """
    Function to save the species counts
    """
    real_df = real_df.copy()
    fake_df = fake_df.copy()

    real_species_counts = real_df['species'].value_counts().reset_index()
    real_species_counts.columns = ['species', 'real_count']
    real_species_counts.to_csv('real_animal_species_counts.csv')

    fake_species_counts = fake_df['species'].value_counts().reset_index()
    fake_species_counts.columns = ['species', 'fake_count']
    fake_species_counts.to_csv('fake_animal_species_counts.csv')

    df = pd.merge(real_species_counts, fake_species_counts, on='species', how='outer').fillna(0)
    df['total_count'] = df['real_count'] + df['fake_count']
    df = df.sort_values(by='total_count')
    df.to_csv('species_balance_counts.csv')

def visualize_species_distribution(real_df, fake_df):
    real_df = real_df.copy()
    fake_df = fake_df.copy()

    real_df['type'] = 'Real'
    fake_df['type'] = 'Fake'
    df = pd.concat([real_df, fake_df])

    top_species = df['species'].value_counts().head(20).index
    df = df[df['species'].isin(top_species)]

    plt.figure(figsize=(18, 10))
    sb.countplot(data=df, x='species', hue='type', palette={"Real": 'royalblue', 'Fake': 'red'})
    plt.title("Real & Fake image balance graph", fontsize=20, pad=20)
    plt.xlabel("Species", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Species")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    df = load_dataframe()
    count_animal_images(df, os.getenv("CLASS_PATH"))

