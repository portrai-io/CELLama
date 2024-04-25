import scanpy as sc
import numpy as np
from random import randint, choices, sample
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import InputExample
import json
import os
import sys
sys.path.append('..')
from cellama import cell_to_sentence
from _examples_to_json import save_examples_to_json

def preprocess_data(adata_, genename = 'feature_name', n_hvgs=1500):
    print("Preprocessing data...")
    adata= adata_.copy()
    adata.var.index = adata.var[genename]
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
    sc.tl.pca(adata)
    return adata[:, adata.var.highly_variable]


def generate_training_data(adata, sentences, num_samples=10000, top_k=10, similarity_ratio=0.5, 
                           batch_random_sampling=100, obs_features=None):
    '''
    Generates training data by creating pairs of sentences with associated cosine similarity scores,
    modified based on the difference in specified observation features.

    Parameters:
        adata (AnnData): The annotated data matrix which includes PCA data under .obsm['X_pca'].
        sentences (list): List of sentences corresponding to the rows in `adata`.
        num_samples (int): Total number of samples to generate.
        top_k (int): Number of high similarity samples to consider for each base sample.
        similarity_ratio (float): Proportion of the total samples that should be high similarity pairs.
        batch_random_sampling (int): Number of random samples to process in each batch for random pairs.
        obs_features (list): List of observation features to consider when adjusting similarity values.

    Returns:
        list: A list of InputExample objects, each containing a pair of sentences and a similarity label.
    '''
    pca_data = adata.obsm['X_pca']
    num_cells = pca_data.shape[0]
    training_examples = []
    num_similarity_samples = int(num_samples * similarity_ratio)
    num_random_samples = num_samples - num_similarity_samples

    def check_features(idx1, idx2):
        ''' Check if any of the observation features are different between two indices '''
        if obs_features is None:
            return True
        for feature in obs_features:
            if adata.obs[feature].iloc[idx1] != adata.obs[feature].iloc[idx2]:
                return False
        return True

    # Generate high similarity samples
    for _ in range(num_similarity_samples // top_k):
        idx1 = randint(0, num_cells - 1)
        similarities = cosine_similarity([pca_data[idx1]], pca_data)[0]
        top_k_indices = np.argsort(-similarities)[1:top_k+1]  # Skip the self-match at 0 position

        for idx2 in top_k_indices:
            if check_features(idx1, idx2):
                cos_sim = similarities[idx2]
            else:
                cos_sim = 0  # Set similarity to 0 if observation features differ
            example = InputExample(texts=[sentences[idx1], sentences[idx2]], label=float(cos_sim))
            training_examples.append(example)

    # Generate random samples using batch processing
    for _ in range(num_random_samples // batch_random_sampling):
        idx1 = randint(0, num_cells - 1)
        idx2s = choices(range(num_cells), k=batch_random_sampling)
        #idx2s = [idx2 for idx2 in idx2s if idx2 != idx1 and check_features(idx1, idx2)]  # Filter idx2s

        cos_sims = cosine_similarity([pca_data[idx1]], pca_data[idx2s])[0]

        for idx2, cos_sim in zip(idx2s, cos_sims):
            if not check_features(idx1, idx2):
                cos_sim = 0  # Adjust similarity based on observation features
            example = InputExample(texts=[sentences[idx1], sentences[idx2]], label=float(cos_sim))
            training_examples.append(example)

    shuffled_examples = sample(training_examples, len(training_examples))
    return shuffled_examples


def generate_and_save_examples(adata_, num_samples, top_k_values, obs_features_options=[None],  n_hvgs_s=[500,1500,3000],
                               save_file = 'adata_sample_train_examples.json', 
                               genename= 'feature_name', verbose=True, return_examples=False):
    all_examples = []
    for n_hvgs in n_hvgs_s:
        if adata_.shape[1] < n_hvgs: 
            print('WARNING: number of var is smaller than  highly variable genes. Will not generate sentence examples')
            continue
        else:
            adata = preprocess_data(adata_, genename = genename, n_hvgs=n_hvgs)
            for top_k in top_k_values:
                for features in obs_features_options:
                    print('........',n_hvgs, top_k, features ) 
                    
                    sentences = list(cell_to_sentence(adata, top_k, features).values())
                    examples = generate_training_data(adata, sentences, num_samples=num_samples, top_k=top_k, similarity_ratio=0.5, batch_random_sampling=100, obs_features=features)

                    if verbose: 
                        print('----sample examples----')
                        for example in examples[:3]:
                            print(f'Texts: {example.texts}, Label: {example.label}')
                        print('\t\t train_exmaples generated')
                    all_examples.extend(examples)


    if save_file:    
        savedir = './sentence_examples/'
        if not os.path.exists(savedir):  
            os.makedirs(savedir) 
        save_examples_to_json(all_examples, file_path= savedir+save_file)
    if return_examples:
        return all_examples

if __name__ == '__main__':
    file_path = './Sentence/Tabula_sapiens_10subsample_raw_counts.h5ad'
    num_samples = 10000  # Adjust based on the actual number of cells
    top_k_values = [16, 20, 24]
    obs_features_options = [None, ['organ_tissue'], ['method', 'organ_tissue']]
    generate_and_save_examples(file_path, num_samples, top_k_values, obs_features_options)