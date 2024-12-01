import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import seaborn as sns

with open('./All_score.json', encoding = 'utf-8') as f:
    data = json.load(f)


outline_scores = {model_name: [] for model_name in data[0].keys()}

for entry in data:
    for model_name, model_data in entry.items():
        # only the score array, ignore the descriptive text
        outline_scores[model_name].append(model_data[0])


normalized_data = {}
for model, scores in outline_scores.items():
    # Convert to DataFrame for each model and normalize scores
    scores_df = pd.DataFrame(scores)
    scaler = StandardScaler()
    normalized_scores = scaler.fit_transform(scores_df)
    normalized_data[model] = pd.DataFrame(normalized_scores, columns=[f"Dimension_{i+1}" for i in range(scores_df.shape[1])])

# Concatenate all normalized data with model labels for dimensionality reduction
all_normalized_data = pd.concat([df.assign(model=model) for model, df in normalized_data.items()], ignore_index=True)
features = all_normalized_data.drop(columns=['model'])
labels = all_normalized_data['model']


intra_outline_similarities = {}

for outline_df in normalized_data:
    outline_name = outline_df['Outline'].iloc[0]
    vectors = outline_df.drop(columns=['Outline', 'Model']).values
    similarity_matrix = cosine_similarity(vectors)
    # 计算平均组内相似度（忽略对角线，即自身相似度）
    avg_intra_similarity = (np.sum(similarity_matrix) - np.trace(similarity_matrix)) / (similarity_matrix.shape[0] ** 2 - similarity_matrix.shape[0])
    intra_outline_similarities[outline_name] = avg_intra_similarity

# 2. 计算大纲之间的向量相似度
# 首先计算每个大纲的平均向量
outline_averages = {}

for outline_df in normalized_data:
    outline_name = outline_df['Outline'].iloc[0]
    vectors = outline_df.drop(columns=['Outline', 'Model']).values
    outline_average = np.mean(vectors, axis=0)
    outline_averages[outline_name] = outline_average

# 计算大纲之间的cosine similarity
outline_names = list(outline_averages.keys())
outline_vectors = np.array(list(outline_averages.values()))
inter_outline_similarity_matrix = cosine_similarity(outline_vectors)

# 将大纲之间的相似度矩阵转换为 DataFrame 以便查看
inter_outline_similarity_df = pd.DataFrame(inter_outline_similarity_matrix, index=outline_names, columns=outline_names)

# 输出组内和大纲之间的相似度
print("组内对比平均相似度（cosine similarity）：")
for outline_name, similarity in intra_outline_similarities.items():
    print(f"{outline_name}: {similarity:.4f}")

print("\n大纲之间的相似度矩阵：")
print(inter_outline_similarity_df)
# Apply dimensionality reduction with PCA (alternatively MDS or t-SNE)
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(features)

# Alternatively, use MDS
# mds = MDS(n_components=2)
# reduced_data = mds.fit_transform(features)

# Alternatively, use t-SNE
tsne = TSNE(n_components=2)
reduced_data = tsne.fit_transform(features)


plot_data = pd.DataFrame(reduced_data, columns=['Dim1', 'Dim2'])
plot_data['Model'] = labels.values

plt.figure(figsize=(12, 8))  # Adjusted figure size for better spacing
sns.scatterplot(x='Dim1', y='Dim2', hue='Model', data=plot_data, palette='viridis', s=100)
plt.title("Dimensionality Reduction of Outline Scores for Each Model")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Adjust the legend
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)
plt.tight_layout()  # Automatically adjusts the layout to make space for the legend

plt.show()

