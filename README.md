# CELLama

<img src="https://github.com/portrai-io/CELLama/assets/103564171/f0211b49-2c8d-45a7-a223-b323c21c3ac1" style="width: 250px;">

---

## Cell Embedding Leveraging Language Model Ability

Developed by Portrai, Inc.

### 🥅 Goal
- The goal is to create a dataset and modality-agnostic cell embedding function using a universal cell embedder. This project leverages a language model to enhance embedding efficiency and evaluates the performance of this approach.

### :microscope: Background
- Cells are classified based on their gene expression, and when testing perturbations, an integration method is necessary. With numerous experiments across different modalities, different samples, a method to integrate this data is crucial. 
We have used embeddings to ensure cells occupy the same space, traditionally done through direct dataset pairing (like scVI, Harmony, Seurat).

- Integration attempts to map cells by proximity, thus, the core challenge is how effectively cells can be embedded in the same space. This has led to the sudden rise of developing a 'universal' cell-specific gene expression embedding as a 'Foundation model'.

- Models such as scGPT, Geneformer, etc. fall under this category. However, the crux is focusing on how gene expression values of each cell are integrated across batches and modalities, which vary significantly, making universal application difficult: more flexibility is needed!

- The hypothesis here is to use gene ranks as well as cell observations to integrate and various features under gene expression data of cells by creating a 'language description' of each cell related to its characteristics. By utilizing the 'sentence-transformer' without additional training (or additional training after generation sentences), we can create a cell-to-sentence function that describes cells and embed them using the sentence transformer. This can be varied from observational descriptions as well as 'gene ranks'.

### :microscope: Implementation
- It utilizes pre-trained sentence transformers to convert cell gene expressions into descriptive sentences, which are then embedded to create a universal embedding space for cells. Also, CELLama covers finetuning (even more, large-sized data-based retraining) to achieve better performance on specific dataset.

### :mag: Key Features
- **Modality Agnostic**: Works across different types of cell gene expression data.
- **Universal Embedding**: Facilitates better integration and comparison across diverse datasets.
- **No Additional Training**: Leverages existing language models, reducing the need for specialized training and simplifying implementation.
- **Fine Tuning**: By generating sentences from cells (gene expression and their metadata), CELLama embedding model can be fine tuned.
- **Spatial Context**: As one of metadata, spatial transcriptomics can provide niche information, such as nearest cells for each cell. By generating sentences with this niche cell information, cell subclustering and characterization considering spatial context are possible. 

---

## Getting Started
To get started with CELLama, download the project files and use pip install. 

#### Prerequisites
- Python (>3.8)
- Run the following command in your terminal
  
```bash
pip install git+https://github.com/portrai-io/CELLama.git
```

- Alternatively, follow these steps:

```bash
### Successful on Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-105-generic x86_64)

conda create --name cellama2 python=3.9
conda activate cellama2
python --version
# Python 3.9.19
git clone https://github.com/portrai-io/CELLama.git 
cd CELLama
pip install -r requirements.txt
pip install -e .

### OPTIONAL
pip install ipykernel jupyter jupyterlab 
python -m ipykernel install --user --name CELLama --display-name CELLama
cd
jupyter lab
```

---

## License
For more information, see the `LICENSE` file included with the project.

---

## Contact
- Citation: 
Choi, H., Park, J., Kim, S., Kim, J., Lee, D., Bae, S., Shin, H., & Lee, D. (2024). CELLama: Foundation model for single cell and spatial transcriptomics by cell embedding leveraging language model abilities. bioRxiv. https://doi.org/10.1101/2024.05.08.593094
- webpage: www.portrai.io
- Portrai: contact@portrai.io ; Hongyoon Choi: chy@portrai.io
