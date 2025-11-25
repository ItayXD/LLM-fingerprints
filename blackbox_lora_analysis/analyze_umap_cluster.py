import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform

# PERFORM U_MAP CLUSTERING ANALYSIS

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
file_list = [
    "/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM_Fingerprint_LoRA/LLAMA_1B_Instruct_LoRA_Claude_3_5_Haiku/adapter_model.safetensors",
    "/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM_Fingerprint_LoRA/LLAMA_1B_Instruct_LoRA_Claude_4_5_Sonnet/adapter_model.safetensors",
    "/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM_Fingerprint_LoRA/LLAMA_1B_Instruct_LoRA_GPT_3_5_Turbo_0125/adapter_model.safetensors",
    "/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM_Fingerprint_LoRA/LLAMA_1B_Instruct_LoRA_GPT_3_5_Turbo_1106/adapter_model.safetensors",
    "/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM_Fingerprint_LoRA/LLAMA_1B_Instruct_LoRA_GPT_4_1/adapter_model.safetensors",
    "/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM_Fingerprint_LoRA/LLAMA_1B_Instruct_LoRA_GPT_4_1_2025_04_14/adapter_model.safetensors",
    "/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM_Fingerprint_LoRA/LLAMA_1B_Instruct_LoRA_Medical/adapter_model.safetensors"
]

# Extract short names for visualization
names = [
    "Claude 3.5 Haiku",
    "Claude 4.5 Sonnet", 
    "GPT-3.5 (0125)",
    "GPT-3.5 (1106)",
    "GPT-4.1",
    "GPT-4.1 (Apr)",
    "Medical"
]

# Color palette for consistent visualization
colors = sns.color_palette("husl", len(names))


# ---------------------------------------------------------
# STEP 1: LOAD AND FLATTEN LoRA WEIGHTS
# ---------------------------------------------------------
print("="*60)
print("STEP 1: Loading LoRA adapters...")
print("="*60)

def load_lora_vector(path):
    """
    Load a LoRA adapter from .safetensors file and flatten all 
    parameters into a single 1D vector.
    """
    state_dict = load_file(path)
    tensors = []
    
    for key, value in state_dict.items():
        # Convert to float32 if needed (handles float16/bfloat16)
        if value.dtype != torch.float32:
            value = value.float()
        # Flatten each parameter tensor and collect
        tensors.append(value.reshape(-1))
    
    # Concatenate all parameters into one long vector
    flat_vector = torch.cat(tensors)
    return flat_vector


# Load all adapters
vectors = []
for i, path in enumerate(file_list):
    print(f"Loading {names[i]}...")
    vec = load_lora_vector(path)
    vectors.append(vec)
    print(f"  → Parameter count: {vec.shape[0]:,}")

# Stack into a matrix: [N_models × N_parameters]
param_matrix = torch.stack(vectors).cpu().numpy()
print(f"\nTotal shape: {param_matrix.shape}")
print(f"  {param_matrix.shape[0]} models")
print(f"  {param_matrix.shape[1]:,} parameters each")


# ---------------------------------------------------------
# STEP 2: NORMALIZE THE VECTORS
# ---------------------------------------------------------
print("\n" + "="*60)
print("STEP 2: Normalizing parameter vectors...")
print("="*60)

# L2 normalization (unit vectors)
param_matrix_norm = param_matrix / np.linalg.norm(param_matrix, axis=1, keepdims=True)

print("Normalization complete.")
print(f"Row norms after normalization: {np.linalg.norm(param_matrix_norm, axis=1)}")


# ---------------------------------------------------------
# STEP 3: APPLY UMAP FOR DIMENSIONALITY REDUCTION
# ---------------------------------------------------------
print("\n" + "="*60)
print("STEP 3: Running UMAP dimensionality reduction...")
print("="*60)

# Create UMAP reducer
reducer = umap.UMAP(
    n_neighbors=5,        
    min_dist=0.1,         
    n_components=2,       
    metric='euclidean',   
    random_state=42       
)

# Fit and transform
print("Fitting UMAP (this may take 10-30 seconds)...")
embedding_2d = reducer.fit_transform(param_matrix_norm)

print("UMAP complete!")
print(f"Embedded shape: {embedding_2d.shape}")
print(f"Embedding X range: [{embedding_2d[:, 0].min():.2f}, {embedding_2d[:, 0].max():.2f}]")
print(f"Embedding Y range: [{embedding_2d[:, 1].min():.2f}, {embedding_2d[:, 1].max():.2f}]")


# ---------------------------------------------------------
# STEP 4: CLUSTERING AND EVALUATION
# ---------------------------------------------------------
print("\n" + "="*60)
print("STEP 4: Clustering analysis with multiple K values...")
print("="*60)

"""
Since we have only 7 samples, we'll test different numbers of clusters
to see which gives the best separation. This helps understand the 
natural grouping structure.
"""

# Ground truth labels (we know which model each adapter came from)
true_labels = np.arange(len(names))

# Test different numbers of clusters
k_values = [2, 3, 4, 5, 6]  # Can't use K=7 for silhouette score
best_k = None
best_score = -1
clustering_results = {}

print("\nTesting different numbers of clusters:")
print("-" * 60)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(embedding_2d)
    
    # Compute silhouette score (only valid when k < n_samples)
    sil_score = silhouette_score(embedding_2d, labels)
    
    # Davies-Bouldin score (lower is better)
    db_score = davies_bouldin_score(embedding_2d, labels)
    
    clustering_results[k] = {
        'labels': labels,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'centers': kmeans.cluster_centers_
    }
    
    print(f"K={k}:")
    print(f"  Silhouette Score:     {sil_score:.3f} (higher is better, >0.5 is good)")
    print(f"  Davies-Bouldin Score: {db_score:.3f} (lower is better)")
    
    if sil_score > best_score:
        best_score = sil_score
        best_k = k

print(f"\n✓ Best K based on silhouette score: K={best_k} (score: {best_score:.3f})")

# Use best K for visualization
best_labels = clustering_results[best_k]['labels']
best_centers = clustering_results[best_k]['centers']

# Also compute K=7 for comparison with ground truth (no silhouette)
kmeans_7 = KMeans(n_clusters=7, random_state=42, n_init=20)
labels_7 = kmeans_7.fit_predict(embedding_2d)
ari_score = adjusted_rand_score(true_labels, labels_7)

print(f"\nK=7 (one cluster per model):")
print(f"  Adjusted Rand Index: {ari_score:.3f}")
print(f"    (1.0 = perfect match, 0.0 = random)")


# ---------------------------------------------------------
# STEP 5: COMPREHENSIVE VISUALIZATION
# ---------------------------------------------------------
print("\n" + "="*60)
print("STEP 5: Creating visualizations...")
print("="*60)

# Create main figure with 4 subplots
fig = plt.figure(figsize=(20, 10))

# ------------------- PLOT 1: Ground Truth Labels -------------------
ax1 = plt.subplot(2, 3, 1)
for i, (x, y) in enumerate(embedding_2d):
    ax1.scatter(x, y, c=[colors[i]], s=400, alpha=0.8, 
               edgecolors='black', linewidth=2.5)
    ax1.text(x, y, str(i), fontsize=11, ha='center', va='center',
            weight='bold', color='white')

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=colors[i], markersize=10, 
                             label=f"{i}: {names[i]}")
                  for i in range(len(names))]
ax1.legend(handles=legend_elements, loc='best', fontsize=8)

ax1.set_title('UMAP Projection\n(True Model Labels)', 
             fontsize=13, weight='bold')
ax1.set_xlabel('UMAP Dimension 1', fontsize=11)
ax1.set_ylabel('UMAP Dimension 2', fontsize=11)
ax1.grid(True, alpha=0.3)

# ------------------- PLOT 2: Best K Clustering -------------------
ax2 = plt.subplot(2, 3, 2)

# Use distinct colors for predicted clusters
cluster_palette = sns.color_palette("Set2", best_k)

for i, (x, y) in enumerate(embedding_2d):
    cluster_color = cluster_palette[best_labels[i]]
    ax2.scatter(x, y, c=[cluster_color], s=400, alpha=0.8,
               edgecolors='black', linewidth=2.5)
    ax2.text(x, y, str(best_labels[i]), fontsize=11, 
            ha='center', va='center', weight='bold', color='white')

# Plot cluster centers
ax2.scatter(best_centers[:, 0], best_centers[:, 1], c='red', s=600, 
           alpha=0.6, marker='X', edgecolors='black', linewidth=3,
           label='Cluster Centers', zorder=10)

ax2.set_title(f'K-Means Clustering (K={best_k})\nSilhouette: {best_score:.3f}', 
             fontsize=13, weight='bold')
ax2.set_xlabel('UMAP Dimension 1', fontsize=11)
ax2.set_ylabel('UMAP Dimension 2', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ------------------- PLOT 3: Pairwise Distances Heatmap -------------------
ax3 = plt.subplot(2, 3, 3)

# Compute pairwise Euclidean distances in UMAP space
distances = squareform(pdist(embedding_2d, metric='euclidean'))

# Create heatmap
im = ax3.imshow(distances, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(len(names)))
ax3.set_yticks(range(len(names)))
short_names = [n.replace(' ', '\n') for n in names]
ax3.set_xticklabels(range(len(names)), fontsize=10)
ax3.set_yticklabels(range(len(names)), fontsize=10)
ax3.set_title('Pairwise Distances\nin UMAP Space', 
             fontsize=13, weight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Euclidean Distance', fontsize=10)

# Annotate with distances
for i in range(len(names)):
    for j in range(len(names)):
        color = 'white' if distances[i, j] > distances.max()/2 else 'black'
        text = ax3.text(j, i, f'{distances[i, j]:.2f}',
                       ha="center", va="center", color=color, 
                       fontsize=9, weight='bold')

# ------------------- PLOT 4: Elbow Plot -------------------
ax4 = plt.subplot(2, 3, 4)

k_range = list(clustering_results.keys())
silhouette_scores = [clustering_results[k]['silhouette'] for k in k_range]

ax4.plot(k_range, silhouette_scores, marker='o', linewidth=2, 
         markersize=10, color='steelblue')
ax4.axvline(best_k, color='red', linestyle='--', linewidth=2, 
           label=f'Best K={best_k}')
ax4.set_xlabel('Number of Clusters (K)', fontsize=11)
ax4.set_ylabel('Silhouette Score', fontsize=11)
ax4.set_title('Silhouette Score vs. K\n(Higher is Better)', 
             fontsize=13, weight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xticks(k_range)

# ------------------- PLOT 5: Davies-Bouldin Plot -------------------
ax5 = plt.subplot(2, 3, 5)

db_scores = [clustering_results[k]['davies_bouldin'] for k in k_range]

ax5.plot(k_range, db_scores, marker='s', linewidth=2, 
         markersize=10, color='coral')
ax5.set_xlabel('Number of Clusters (K)', fontsize=11)
ax5.set_ylabel('Davies-Bouldin Score', fontsize=11)
ax5.set_title('Davies-Bouldin Index vs. K\n(Lower is Better)', 
             fontsize=13, weight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xticks(k_range)

# ------------------- PLOT 6: Cluster Assignment Table -------------------
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Create table showing cluster assignments for best K
table_data = []
for i in range(len(names)):
    table_data.append([
        str(i),
        names[i][:20],  # Truncate long names
        str(best_labels[i])
    ])

table = ax6.table(cellText=table_data, 
                 colLabels=['ID', 'Model', f'Cluster\n(K={best_k})'],
                 cellLoc='left',
                 loc='center',
                 colWidths=[0.1, 0.6, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code by cluster
for i in range(1, len(names) + 1):
    cluster_id = best_labels[i-1]
    color = cluster_palette[cluster_id]
    table[(i, 2)].set_facecolor(color)
    table[(i, 2)].set_text_props(weight='bold')

ax6.set_title('Cluster Assignments', fontsize=13, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('umap_clustering_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: umap_clustering_analysis.png")
plt.show()


# ---------------------------------------------------------
# STEP 6: DETAILED ANALYSIS OUTPUT
# ---------------------------------------------------------
print("\n" + "="*60)
print("STEP 6: Detailed Analysis")
print("="*60)

# Show cluster memberships for best K
print(f"\n🔍 Cluster assignments (K={best_k}):")
print("-" * 60)
clusters_dict = {}
for i in range(len(names)):
    cluster_id = best_labels[i]
    if cluster_id not in clusters_dict:
        clusters_dict[cluster_id] = []
    clusters_dict[cluster_id].append((i, names[i]))

for cluster_id in sorted(clusters_dict.keys()):
    print(f"\nCluster {cluster_id}:")
    for model_id, model_name in clusters_dict[cluster_id]:
        print(f"  [{model_id}] {model_name}")

# Find which models are closest in UMAP space
print("\n📏 Closest model pairs in UMAP space:")
print("-" * 60)
upper_tri = np.triu_indices(len(names), k=1)
dist_pairs = [(distances[i, j], i, j, names[i], names[j]) 
              for i, j in zip(*upper_tri)]
dist_pairs.sort()

for dist, i, j, name1, name2 in dist_pairs[:5]:
    print(f"  [{i}] {name1:25s} ↔ [{j}] {name2:25s}: {dist:.3f}")

# Find which models are farthest
print("\n📏 Most distant model pairs in UMAP space:")
print("-" * 60)
for dist, i, j, name1, name2 in dist_pairs[-5:]:
    print(f"  [{i}] {name1:25s} ↔ [{j}] {name2:25s}: {dist:.3f}")

# Compute average within-cluster vs between-cluster distances
print("\n📊 Cluster quality metrics:")
print("-" * 60)
within_cluster_dists = []
between_cluster_dists = []

for i in range(len(names)):
    for j in range(i+1, len(names)):
        if best_labels[i] == best_labels[j]:
            within_cluster_dists.append(distances[i, j])
        else:
            between_cluster_dists.append(distances[i, j])

if len(within_cluster_dists) > 0:
    print(f"Average within-cluster distance:  {np.mean(within_cluster_dists):.3f}")
    print(f"Average between-cluster distance: {np.mean(between_cluster_dists):.3f}")
    ratio = np.mean(between_cluster_dists) / np.mean(within_cluster_dists)
    print(f"Separation ratio:                 {ratio:.3f}x")
    print(f"  (Higher ratio = better separation)")
else:
    print("All points in separate clusters (K=N)")


# ---------------------------------------------------------
# STEP 7: 3D UMAP VISUALIZATION
# ---------------------------------------------------------
print("\n" + "="*60)
print("STEP 7: Creating 3D UMAP visualization...")
print("="*60)

# Compute 3D embedding
reducer_3d = umap.UMAP(
    n_neighbors=5,
    min_dist=0.1,
    n_components=3,  
    metric='euclidean',
    random_state=42
)

embedding_3d = reducer_3d.fit_transform(param_matrix_norm)

# Create 3D plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for i, (x, y, z) in enumerate(embedding_3d):
    ax.scatter(x, y, z, c=[colors[i]], s=400, alpha=0.8,
              edgecolors='black', linewidth=2)
    ax.text(x, y, z, f" {i}", fontsize=10, weight='bold')

# Add legend
for i, name in enumerate(names):
    ax.scatter([], [], [], c=[colors[i]], s=100, label=f"{i}: {name}")

ax.set_xlabel('UMAP Dim 1', fontsize=11, weight='bold')
ax.set_ylabel('UMAP Dim 2', fontsize=11, weight='bold')
ax.set_zlabel('UMAP Dim 3', fontsize=11, weight='bold')
ax.set_title('3D UMAP Projection of LoRA Adapters', 
            fontsize=14, weight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

plt.tight_layout()
plt.savefig('umap_3d_clustering.png', dpi=300, bbox_inches='tight')
print("✓ Saved: umap_3d_clustering.png")
plt.show()

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print(f"\nKey Findings:")
print(f"  • Optimal number of clusters: K={best_k}")
print(f"  • Best silhouette score: {best_score:.3f}")
print(f"  • Separation quality: {'Good' if best_score > 0.5 else 'Moderate' if best_score > 0.25 else 'Weak'}")