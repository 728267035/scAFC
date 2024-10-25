
Requirements:
Python --- 3.7.0
pytorch -- 1.8.1+cu111
Numpy --- 1.21.6
Pandas --- 1.3.5
Scipy --- 1.7.3
Sklearn --- 1.0.2

Datasets:
Due to GitHub space limitations, the complete data files cannot be hosted directly on it. The detailed data files have been uploaded to OneDrive; please access them via the following link: OneDrive Link

Files:
Data Preprocessing: After obtaining the scRNA-seq data, use scanpy_filter.py to preprocess the gene expression data.
Graph Generation: Run the calcu_graph.py file to generate the required input graph and store it in the graph folder.
Pre-training: To improve training outcomes, we conducted pre-training. Run preae.py to generate a pre-trained model and save the .pkl file in the model folder.
Training: Run the scAFC.py file to train the final model.
