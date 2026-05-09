import tensorflow as tf
from tensorflow.keras import layers, models, Model, regularizers
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

def create_base_network(input_shape, embedding_dim=128):
    """
    Creates the base embedding network (MLP) with Batch Normalization
    and L2 Regularization, which is standard for tabular ZSL.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1
    x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Layer 2
    x = layers.Dense(128, use_bias=False, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output Layer (L2 Normalized Embeddings)
    # No batch norm on output usually for metric learning, just direct normalization
    x = layers.Dense(embedding_dim, activation=None)(x)
    embeddings = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(x)
    
    return Model(inputs, embeddings, name="BaseEmbeddingNetwork")

def get_triplet_model(input_shape, embedding_dim=128):
    """
    Constructs the model for Triplet Training.
    Note: For TripletSemiHardLoss, we typically just need the embeddings output 
    and pass (y_true, y_pred) to the loss function where y_pred is the embedding.
    """
    base_model = create_base_network(input_shape, embedding_dim)
    
    # We can wrap it if needed, but standard training loop usually applies 
    # loss on the embeddings directly.
    return base_model

class HybridClusterModel:
    """
    Manages the prototype generation and ZSL prediction.
    """
    def __init__(self, embedding_model, method='kmeans', n_clusters=1):
        """
        Args:
            embedding_model: Trained tf.keras.Model
            method: 'kmeans', 'gmm', or 'agglomerative'
            n_clusters: Number of prototypes per class (reviewer comment addressed here)
        """
        self.embedding_model = embedding_model
        self.method = method
        self.n_clusters = n_clusters
        self.prototypes = {} # Dict[class_label] -> list of prototypes (vectors)
        self.class_mapping = {} # Just to keep track of classes
        
    def fit_prototypes(self, X_seen, y_seen):
        """
        Generates prototypes for each SEEN class using the trained embedding model.
        IN ZSL settings, we might use attribute vectors, but here we use 
        prototypes derived from the 'Support Set' (which in some definitions 
        is the training data, for Generalized ZSL).
        
        For strict ZSL on Unseen classes, we assume we have a small set of 
        labeled samples (Support Set) for those unseen classes at test time 
        (Few-Shot scenario) OR we rely on semantic attributes (which are not present here?).
        
        CRITICAL NOTE: The user's ZSL approach in the original paper/code implies 
        using prototypes generated from data. If there are no side-information/attributes, 
        this effectively becomes Few-Shot Learning or Metric Learning 
        where we classify queries based on support sets.
        
        We will generate prototypes for whatever classes are passed in (Support Set).
        """
        print(f"Generating prototypes using method: {self.method} with n_clusters={self.n_clusters}")
        
        # Get embeddings
        embeddings = self.embedding_model.predict(X_seen, verbose=0)
        
        unique_classes = np.unique(y_seen)
        self.prototypes = {}
        
        for cls in unique_classes:
            cls_indices = np.where(y_seen == cls)[0]
            cls_embeddings = embeddings[cls_indices]
            
            if self.method == 'kmeans':
                # Use KMeans to find centroids
                # Handle case where samples < n_clusters
                n = min(len(cls_embeddings), self.n_clusters)
                if n < self.n_clusters:
                    print(f"Warning: Class {cls} has {len(cls_embeddings)} samples, < n_clusters {self.n_clusters}. Using {n}.")
                
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                kmeans.fit(cls_embeddings)
                self.prototypes[cls] = kmeans.cluster_centers_
                
            elif self.method == 'gmm':
                n = min(len(cls_embeddings), self.n_clusters)
                gmm = GaussianMixture(n_components=n, random_state=42)
                gmm.fit(cls_embeddings)
                self.prototypes[cls] = gmm.means_
                
            elif self.method == 'agglomerative':
                 # Agglomerative doesn't give centroids directly easily in sklearn API 
                 # in the same way, but we can compute mean of clusters.
                 # For simplicity/robustness, we might fallback to mean if n=1 or similar to KMeans logic
                 # Actually, let's just stick to Mean if n_clusters=1, else KMeans is better than Agglo for prototypes usually.
                 # Reproducing original logic:
                 n = min(len(cls_embeddings), self.n_clusters)
                 agg = AgglomerativeClustering(n_clusters=n)
                 labels = agg.fit_predict(cls_embeddings)
                 # Compute centroids manually
                 centers = []
                 for i in range(n):
                     cluster_points = cls_embeddings[labels == i]
                     centers.append(cluster_points.mean(axis=0))
                 self.prototypes[cls] = np.array(centers)
                 
    def predict(self, X_query):
        """
        Predicts class for query samples based on nearest existing prototype.
        """
        query_embeddings = self.embedding_model.predict(X_query, verbose=0)
        predictions = []
        
        # We need to flatten our prototypes for efficient search
        # Or iterate. For clarity, let's iterate.
        
        for q_emb in query_embeddings:
            best_sim = -1.0 # Cosine similarity ranges [-1, 1]
            best_cls = None
            
            for cls, protos in self.prototypes.items():
                # Check against all prototypes for this class
                for proto in protos:
                    # Cosine Similarity
                    # vectors are already L2 normalized by the model (supposedly), but let's be safe
                    # sim = (A . B) / (|A|*|B|)
                    # precise calc:
                    sim = np.dot(q_emb, proto) / (np.linalg.norm(q_emb) * np.linalg.norm(proto) + 1e-9)
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_cls = cls
            
            predictions.append(best_cls)
            
        return np.array(predictions)
