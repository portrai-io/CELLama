# CELLama

<img src="https://github.com/portrai-io/CELLama/assets/103564171/f0211b49-2c8d-45a7-a223-b323c21c3ac1" style="width: 250px;">

## Cell Embedding Leveraging Language Model without Additional Training by Portrai

### Goal
The goal is to create a dataset and modality-agnostic cell embedding function using a universal cell embedder. This project leverages a language model to enhance embedding efficiency and evaluates the performance of this approach.

### :microscope: Background
Cells are classified based on their gene expression, and when testing perturbations, an integration method is necessary. With numerous experiments across different modalities, a method to integrate this data is crucial. We use embeddings to ensure cells occupy the same space, traditionally done through direct dataset pairing (like scVI, Harmony, Seurat).

Integration attempts to map cells by proximity, thus, the core challenge is how effectively cells can be embedded in the same space. This has led to the sudden rise of developing a 'universal' cell-specific gene expression embedding as a 'Foundation model'.
Models such as scGPT, Geneformer, etc. fall under this category. However, the crux is focusing on how gene expression values of each cell are integrated across batches and modalities, which vary significantly, making universal application difficult: more flexibility is needed!

The hypothesis here is to use gene ranks as well as cell observations to integrate and various features under gene expression data of cells by creating a 'language description' of each cell related to its characteristics. By utilizing the 'sentence-transformer' without additional training (or additional training after generation sentences), we can create a cell-to-sentence function that describes cells and embed them using the sentence transformer. This can be varied from observational descriptions as well as 'gene ranks'.

### :microscope: Implementation
This project does not require additional training of models. Instead, it utilizes pre-trained sentence transformers to convert cell gene expressions into descriptive sentences, which are then embedded to create a universal embedding space for cells.

### :mag: Key Benefits
- **Modality Agnostic**: Works across different types of cell gene expression data.
- **Universal Embedding**: Facilitates better integration and comparison across diverse datasets.
- **No Additional Training**: Leverages existing language models, reducing the need for specialized training and simplifying implementation.

## Getting Started
To get started with CELLama, download the project files and follow the setup instructions in the documentation. This will guide you through the process of using the CELLama embedding system in your research or application.

## License
This project is licensed for non-commercial use only. For more information, see the `LICENSE` file included with the project.

## Contact
- Project Link: [GitHub - CELLama](https://github.com/CELLama)
- webpage: portrai.io
- Hongyoon Choi; chy@portrai.io
