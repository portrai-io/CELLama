import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

def find_and_order_neighbors(adata, cellname, n_neighs=5, radius=None, nearest_cells_name= 'nearest_cells'):
    """
    Find and order neighbors within a specified radius or by a fixed number of nearest neighbors.
    
    Parameters:
        adata (AnnData): Annotated data matrix.
        cellname (str): Column name in adata.obs to use for identifying cell identifiers.
        radius (int, optional): Radius within which neighbors are considered. Default is 10.
        n_neighs (int, optional): Number of nearest neighbors if not using radius.
    Returns:
        adata (AnnData): Updated Annotated data matrix with 'nearest_cells' added.
    """
    if radius:
        if n_neighs:
            print('Warning: radius is not None, use radius')
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type='generic')
    else:
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type='generic')

    spatial_distances = adata.obsp['spatial_distances']
    nearest_cells = []

    for i in range(spatial_distances.shape[0]):
        if i % 10000 == 0:
            print(f'Niche cells : {i}/{spatial_distances.shape[0]} for calculation')
        row = spatial_distances[i]
        indices = row.indices
        distances = row.data
        sorted_indices = indices[np.argsort(distances)]
        cell_ids = adata.obs[cellname].iloc[sorted_indices].tolist()
        nearest_cells.append(','.join(cell_ids))

    adata.obs[nearest_cells_name] = nearest_cells
    return adata

def calculate_nichecells_summary(adata, clustername, nichename='nearest_cells', top_number=None):
    """
    Calculate a summary of nearest cells with an option to return only top results.

    Parameters:
        adata (AnnData): Annotated data matrix.
        clustername (str): Column name in adata.obs representing cluster assignment.
        nichename (str): Column name in adata.obs where nearest cells are stored.
        top_number (int, optional): Number of top rows to return in the summary.
    Returns:
        dict: A dictionary with cluster names as keys and normalized cell type counts as values.
    """
    df_aa = pd.crosstab(adata.obs[nichename], adata.obs[clustername])
    summary_nichecells = {}

    for column in df_aa.columns:
        sorted_series = df_aa[column].sort_values(ascending=False)
        top_entries = sorted_series.head(top_number) if top_number else sorted_series

        total_cells = []
        cell_counts = 0
        for row in top_entries.index:
            cells = row.split(',')
            cell_count = df_aa.loc[row, column]
            total_cells_row = cell_count * cells
            total_cells.extend(total_cells_row)
            cell_counts += cell_count

        summary_nichecells[column] = pd.Series(total_cells).value_counts() / cell_counts
    summary_nichecells= pd.DataFrame(summary_nichecells)
    return summary_nichecells
