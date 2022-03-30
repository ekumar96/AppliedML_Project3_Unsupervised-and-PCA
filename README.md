# Project 3 - Spring 2022

Applied ML Spring 2022 Project 3: Imbalanced Data, Unsupervised Learning, and Dimensionality Reduction

Eshan Kumar, ek3227

This project consists of Two parts. In the first part of the project, I work on a heavily imbalanced dataset of the performance of credit card fraud projected onto its first 28 Principal components. The aim is to detect a mere 492 fraudulent transactions from 284,807 transactions in total. The data is a Kaggle dataset, and can be found [here](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)

# Part 1 - Imbalanced Data
In the first part of the project, I try various methods to deal with heavily imbalanced data and evaluate metrics to compare them.

### Data Examination and Cleaning
First, I examine the data in order to see the imabalance. I remove irrelevant features, such as `Time`, and scale `Amount` by log. I split the data into training and test sets, ensuring that I do a stratified split, which is important for imbalanced datasets. I then check and ensure that both training and test sets have the same percentage of positive and negative data.   

### Data Preprocessing
Next, I normalize the data using StandardScaler().

### Default Logistic Regression
I fit the default Logistic Regression model with the imbalanced data, and get metrics (AUC, Average Precision) to evaluate performance. This allows us to have a baseline to compare the future methods to. 

### Random Undersampling
I randomly undersample the majority class, resulting in a large amount of data loss, and fit a default Logistic Regression model with this balanced data, getting metrics (AUC, Average Precision) to evaluate performance. 

### Random Oversampling
I randomly oversample the minority class, resulting in overfitting to the minority data, and fit a default Logistic Regression model with this balanced data, getting metrics (AUC, Average Precision) to evaluate performance. 

### Synthetic Minority Oversampling Technique (SMOTE)
I use SMOTE to generate synthetic data for the minority class (based on the neighbors of random points and the lines between them), then randomly oversample from this synthetic data, which may result in overfitting to the minority data, and fit a default Logistic Regression model with this balanced data, getting metrics (AUC, Average Precision) to evaluate performance. 

### Technique Evaluation (Confusion Matrix, PR Curve, ROC Curve)
I plot confusion matrices, ROC curves (plotting false positive rate vs true positive rate as the threhold is changed), and PR curves (plotting Precision vs. Recall as the threshold is changed) on the test set for the above methods to compare their performance. SMOTE and Random oversampling appear to perform the best.  

### Logistic Regression with Class Weights
I train a logistic regression model with class weights, which weights the data from the minority class more heavily so that it is learned better. The data is weighted according to its rate of appearance in the data. The average precision of this method is low, but the AUC is high, and the confusion matrix, PR curve, and ROC curve show good results for this method.


# Part 2 - Unsupervised Learning and Dimensionality Reduction
In this part, we apply unsupervised learning approaches to a problem in computational biology. Specifically, we analyze single-cell genomic sequencing data. A common challenge of genomic datasets is their high-dimensionality: a single observation (a cell, in the case of single-cell data) may have tens of thousands of gene expression features. 

We will be working with a single-cell RNASeq dataset of mouse brain cells. Each entry in the matrix is a normalized gene expression count - a higher value means that the gene is expressed more in that cell. The dataset has been pre-processed using various quality control and normalization methods for single-cell data.

### PCA
We use PCA to project the data (with 18585 features) onto its first 50 principal components. We then evaluate the amount of variance that is explained in each of the first principal components. For the first principal component, we find the top 10 weights, which correspond to the genes that are deemed the most important in seperating cells (explaining variance) for the first principal component. 

We then plot the data projected onto the its first two principal components, and color the data according to various categorical variables in the metadata. We color the data according to Cell Ontology Class (cell type: Neuron, Astrocyte, etc), Subtissue Categories (location: Hippocampus, Cortex), Mouse Sex (M/F), and Mouse ID (which mouse cells were taken from). We find that clusters in these plots correspond most closely to cell ontology class, or the type of cell. This is good, because it means that given a new set of genes, we can easily tell what type of cell it is, and it also means that there is no significant difference in cells across Mice sex or ID. 

### K-means clustering
While the annotations provide high-level information on cell type (e.g. cell_ontology_class has 7 categories), we may also be interested in finding more granular subtypes of cells. To achieve this, we use K-means clustering to find a large number of clusters in the gene expression dataset. 

We first use PCA to project the data onto it's first 20 principal components: the original gene expression matrix haw over 18,000 noisy features, which is not ideal for clustering. We create a K-means algorithm from scratch which Initializes centers as random points in the data, for each point, finds the center that is the shortest euclidean distance away, and assigns a cluster assignment for this point accordingly, then finally moves the center points to the center of these clusters, and does this in a loop for a given number of iterations. 

We use this algorithm to cluster a synthetic dataset, plotting the dataset with colored clusters, then cluster the gene expression dataset, creating 20, much more granular clusters. 

### t-distributed Stochastic Neighbor Embedding (t-SNE)
We visualize the data again using t-SNE - a non-linear dimensionality reduction algorithm. t-SNE in this interactive tutorial [here](https://distill.pub/2016/misread-tsne/). 

We perform t-SNE on the first 20 principal components (after applying PCA) to speed up computation and suppress noise. We then plot the data (first 20 principal components) projected onto the first two t-SNE dimensions, colored by their cluster assignments from the previous K-means clustering. 

There are overlaps between points in different clusters in the t-SNE plot because t-SNE does not preserve distances nor density, so the clusters seen in the data projected onto the t-SNE dimensions may not correspond to the clusters found in the higher dimensional data after K-means. Some of the clusters found in t-SNE may simply be artifacts of the t-SNE process, and these cluster shapes can change significantly as the perplexity is changed. Additionally, the original data points are assumed to follow a local Gaussian distribution before t-SNE, which is not the case with this data.

These 20 clusters may correspond to various cell subtypes or cell states. They can be further investigated and mapped to known cell types based on their gene expressions (e.g. using the K-means cluster centers). The clusters may also be used in downstream analysis. For instance, we can monitor how the clusters evolve and interact with each other over time in response to a treatment.
