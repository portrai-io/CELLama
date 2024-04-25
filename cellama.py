from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from _nn_model import SimpleNN, train_model
import torch
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# Generate Sentence from scRNA-seq data
def get_top_genes(adata, n_top=30):
    X = adata.X
    if isinstance(X, np.ndarray):
        X = csr_matrix(X)
    
    # Placeholder dictionary to store the result
    top_genes_dict = {}
    # Iterate over each row (cell) in the matrix
    for index in range(X.shape[0]):
        # Get the row as a dense array to work with argsort
        row_data = X[index].toarray().ravel()
        
        # Get the indices of the top 'n_top' values
        top_indices = np.argsort(-row_data)[:n_top]
        
        # Map indices to gene names
        gene_names = adata.var.index[top_indices].tolist()
        top_genes_dict[index] = gene_names
    return top_genes_dict

def list_to_sentence(top_genes_dict):
    # Create a new dictionary to hold the formatted sentences
    formatted_dict = {}
    for index, genes in top_genes_dict.items():
        # Convert the list of genes to a sentence
        if len(genes) > 1:
            gene_sentence = "Top genes are {}, and {}.".format(", ".join(genes[:-1]), genes[-1])
        elif len(genes) ==0:
            gene_sentence = ""
        else:
            gene_sentence = "Top gene is {}.".format(genes[0])
        formatted_dict[index] = gene_sentence
    
    return formatted_dict

def cell_to_sentence(adata, n_top=30, obs_features=None):
    top_genes_dict = get_top_genes(adata, n_top) 
    formatted_dict = {}
    
    for cell_id, genes in top_genes_dict.items():
        # Start with the gene information
        if len(genes) > 1:
            gene_sentence = "Top genes are {}, and {}.".format(", ".join(genes[:-1]), genes[-1])
        else:
            gene_sentence = "Top gene is {}.".format(genes[0])
        
        # Append information from obs if features are specified
        if obs_features:
            for feature in obs_features:
                if feature in adata.obs:
                    feature_value = str(adata.obs[feature].iloc[cell_id])
                    gene_sentence += " {} of this cell is {}.".format(feature, feature_value)
        
        formatted_dict[cell_id] = gene_sentence
    return formatted_dict

def setup_knn_search(db, n_neighbors=1):
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine')
    knn.fit(db)
    return knn

def find_nearest_cell_types(knn, db_new, cell_types):
    distances, indices = knn.kneighbors(db_new)
    nearest_cell_types = [cell_types[i[0]] for i in indices]
    nearest_cell_distances = [distance[0] for distance in distances]

    return nearest_cell_types, nearest_cell_distances


def lm_cell_embed(adata_, top_k=30, model_name="all-MiniLM-L6-v2",gene_list=None, obs_features=None,
                 return_sentence=True):
    
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    if gene_list is not None:
        adata = adata_[:,gene_list].copy()
    else:
        adata = adata_.copy()
        
    top_genes_sentences = cell_to_sentence(adata, top_k, obs_features)
    sentences = list(top_genes_sentences.values())
    db = embedding_function.embed_documents(sentences)
    emb_res = np.asarray(db)
    
    adata_emb = sc.AnnData(emb_res)
    adata_emb.var= pd.DataFrame(range(emb_res.shape[1]))
    adata_emb.obs = adata.obs
    sc.tl.pca(adata_emb, svd_solver='arpack')
    sc.pp.neighbors(adata_emb, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata_emb)
    
    if return_sentence:
        adata_emb.obs['cell_sentence']= sentences
    return adata_emb
    

def lm_cell_reference_celltyping(adata_ref_, adata_new_, model_name="all-MiniLM-L6-v2",
                                 top_k=30, use_intersect= True, gene_list=None, obs_features = None,
                                 ref_cell = 'Cell_type', new_name='CellType_LM'):
    
    if gene_list is not None and use_intersect:
        raise ValueError("Both gene_list and use_intersect are set. Please choose only one method for gene selection.")

    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

    if gene_list is not None:
        adata_ref = adata_ref_[:,gene_list].copy()
        adata_new = adata_new_[:,gene_list].copy()
    
    elif use_intersect:
        int_genes= [ii for ii in adata_ref_.var.index.tolist() if ii in adata_new_.var.index.tolist()]
        adata_ref=  adata_ref_[:,int_genes].copy()
        adata_new = adata_new_[:,int_genes].copy()
    else:
        adata_ref = adata_ref_.copy()
        adata_new = adata_new_.copy()

    print("Language Model Cell Embedding Is Implemented")

    top_genes_sentences = cell_to_sentence(adata_ref, top_k, obs_features)
    sentences = list(top_genes_sentences.values())
    db = embedding_function.embed_documents(sentences)
    emb_res = np.asarray(db)

    top_genes_sentences_new = cell_to_sentence(adata_new, top_k, obs_features)
    sentences_new = list(top_genes_sentences_new.values())
    db_new = embedding_function.embed_documents(sentences_new)
    #Cell Typing with knn model
    knn_model = setup_knn_search(db)
    nearest_cell_types, nearest_cell_distances = find_nearest_cell_types(knn_model, db_new, adata_ref.obs[ref_cell])
    adata_new.obs[new_name]= nearest_cell_types
    adata_new.obs[new_name+'_distance'] = nearest_cell_distances
    return adata_new


def lm_cell_integrated_embed(adata_ref_, adata_new_, model_name="all-MiniLM-L6-v2",
                            top_k=30, use_intersect= True, gene_list=None, obs_features = None):
    if gene_list is not None and use_intersect:
        raise ValueError("Both gene_list and use_intersect are set. Please choose only one method for gene selection.")

    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

    if gene_list is not None:
        adata_ref = adata_ref_[:,gene_list].copy()
        adata_new = adata_new_[:,gene_list].copy()
    
    elif use_intersect:
        int_genes= [ii for ii in adata_ref_.var.index.tolist() if ii in adata_new_.var.index.tolist()]
        adata_ref=  adata_ref_[:,int_genes].copy()
        adata_new = adata_new_[:,int_genes].copy()
    else:
        adata_ref = adata_ref_.copy()
        adata_new = adata_new_.copy()
        
    print("Language Model Cell Embedding Is Implemented")

    top_genes_sentences = cell_to_sentence(adata_ref, top_k, obs_features)
    sentences = list(top_genes_sentences.values())
    db = embedding_function.embed_documents(sentences)
    emb_res = np.asarray(db)

    top_genes_sentences_new = cell_to_sentence(adata_new, top_k, obs_features)
    sentences_new = list(top_genes_sentences_new.values())
    db_new = embedding_function.embed_documents(sentences_new)
    emb_res_new = np.asarray(db_new)

    adata_emb = sc.AnnData(emb_res)
    adata_emb.var= pd.DataFrame(range(emb_res.shape[1]))
    adata_emb.obs = adata_ref.obs

    adata_emb_new = sc.AnnData(emb_res_new)
    adata_emb_new.var= pd.DataFrame(range(emb_res_new.shape[1]))
    adata_emb_new.obs = adata_new.obs
    #adata_all = adata_emb.concatenate(adata_emb_new, index_unique="-", join="outer")

    return adata_emb, adata_emb_new
        
    
def lm_cell_reference_celltyping_nn(adata_ref_, adata_new_, model_name="all-MiniLM-L6-v2",
                                     top_k=30, use_intersect=True, gene_list=None, obs_features=None,
                                     ref_cell='Cell_type', new_name='CellType_LM', epochs=1000):
    # Filter genes based on gene_list or intersection
    if gene_list is not None and use_intersect:
        raise ValueError("Both gene_list and use_intersect are set. Please choose only one method for gene selection.")

    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    if gene_list is not None:
        adata_ref = adata_ref_[:,gene_list].copy()
        adata_new = adata_new_[:,gene_list].copy()
    
    elif use_intersect:
        int_genes= [ii for ii in adata_ref_.var.index.tolist() if ii in adata_new_.var.index.tolist()]
        adata_ref=  adata_ref_[:,int_genes].copy()
        adata_new = adata_new_[:,int_genes].copy()
    else:
        adata_ref = adata_ref_.copy()
        adata_new = adata_new_.copy()

    print("Language Model Cell Embedding Is Implemented")

    # Embedding
    top_genes_sentences = cell_to_sentence(adata_ref, top_k, obs_features)
    sentences = list(top_genes_sentences.values())
    db = embedding_function.embed_documents(sentences)
    emb_res = np.asarray(db)

    top_genes_sentences_new = cell_to_sentence(adata_new, top_k, obs_features)
    sentences_new = list(top_genes_sentences_new.values())
    db_new = embedding_function.embed_documents(sentences_new)
    emb_res_new = np.asarray(db_new)

    # Prepare for training
    cell_types = pd.Categorical(adata_ref.obs[ref_cell]).codes
    input_dim = emb_res.shape[1]
    output_dim = len(np.unique(cell_types))

    # Create dataset and dataloader
    total_dataset = TensorDataset(torch.FloatTensor(emb_res), torch.LongTensor(cell_types))
    train_loader = DataLoader(total_dataset, batch_size=64, shuffle=True)

    # Initialize and train the neural network
    model = SimpleNN(input_dim, output_dim,latent_dim =256, dropout_rate=0.5)
    trained_model = train_model(model, train_loader, n_epochs=epochs)

    # Predict cell types for new data
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predictions = trained_model(torch.FloatTensor(emb_res_new))
        predicted_labels = torch.argmax(predictions, axis=1)
        adata_new.obs[new_name] = [adata_ref.obs[ref_cell].cat.categories[label] for label in predicted_labels.numpy()]

    return adata_new

#XGBoost
def lm_cell_reference_celltyping_xgb(adata_ref_, adata_new_, model_name="all-MiniLM-L6-v2",
                                     top_k=30, use_intersect=True, gene_list=None, obs_features=None,
                                     ref_cell='Cell_type', new_name='CellType_LM'):
    # Filter genes based on gene_list or intersection
    if gene_list is not None and use_intersect:
        raise ValueError("Both gene_list and use_intersect are set. Please choose only one method for gene selection.")

    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    if gene_list is not None:
        adata_ref = adata_ref_[:,gene_list].copy()
        adata_new = adata_new_[:,gene_list].copy()
    
    elif use_intersect:
        int_genes= [ii for ii in adata_ref_.var.index.tolist() if ii in adata_new_.var.index.tolist()]
        adata_ref=  adata_ref_[:,int_genes].copy()
        adata_new = adata_new_[:,int_genes].copy()
    else:
        adata_ref = adata_ref_.copy()
        adata_new = adata_new_.copy()

    print("Language Model Cell Embedding Is Implemented")

    # Embedding
    top_genes_sentences = cell_to_sentence(adata_ref, top_k, obs_features)
    sentences = list(top_genes_sentences.values())
    db = embedding_function.embed_documents(sentences)
    emb_res = np.asarray(db)

    top_genes_sentences_new = cell_to_sentence(adata_new, top_k, obs_features)
    sentences_new = list(top_genes_sentences_new.values())
    db_new = embedding_function.embed_documents(sentences_new)
    emb_res_new = np.asarray(db_new)

    # Prepare for training
    cell_types = pd.Categorical(adata_ref.obs[ref_cell]).codes

    # Initialize and train XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric='mlogloss',
                         random_state=42)
    #xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    model.fit(emb_res, cell_types)

    # Predict cell types for new data
    predicted_labels = model.predict(emb_res_new)
    adata_new.obs[new_name] = [adata_ref.obs[ref_cell].cat.categories[label] for label in predicted_labels]

    return adata_new

