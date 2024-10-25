# scAFC: Adaptive Fusion Clustering of Single-cell RNA-seq data through Autoencoder and Graph Attention Networks

## Requirements:
- Python --- 3.7.0
- pytorch -- 1.8.1+cu111
- Numpy --- 1.21.6
- Pandas --- 1.3.5
- Scipy --- 1.7.3
- Sklearn --- 1.0.2

## Datasets:
Due to GitHub space limitations, the complete data files cannot be hosted directly on it. The detailed data files have been uploaded to OneDrive; please access them via the following link:https://1drv.ms/u/c/1aeb113afecf8673/Ee0Ms7i59U5Hg4pnks01mOoBDMfex17XMzI_RBiF-2QocQ

## Files:
1. Data Preprocessing: After obtaining the scRNA-seq data, use scanpy_filter.py to preprocess the gene expression data.
2. Graph Generation: Run the calcu_graph.py file to generate the required input graph and store it in the graph folder.
3. Pre-training: To improve training outcomes, we conducted pre-training. Run preae.py to generate a pre-trained model and save the .pkl file in the model folder.
4. Training: Run the scAFC.py file to train the final model.
