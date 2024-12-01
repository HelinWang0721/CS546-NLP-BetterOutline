import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load JSON data
with open('./All_score.json', encoding='utf-8') as f:
    data = json.load(f)

# Initialize storage to separate each outline's scores across models
outline_data = []

# Separate each outline's scores across models
for outline_idx, entry in enumerate(data):
    outline_scores = []
    for model_name, model_data in entry.items():
        scores = model_data[0]
        outline_scores.append(scores)
    outline_data.append(pd.DataFrame(outline_scores, index=entry.keys()))

# Normalize each model's scores for each outline separately
normalized_outlines = []
for outline_idx, df in enumerate(outline_data):
    scaler = StandardScaler()
    normalized_scores = scaler.fit_transform(df.T).T  # Normalize each outline independently
    outline_df = pd.DataFrame(normalized_scores, columns=[f"Dimension_{i+1}" for i in range(df.shape[1])])
    outline_df['Outline'] = f'Outline_{outline_idx + 1}'  # Label each outline
    outline_df['Model'] = df.index
    normalized_outlines.append(outline_df)

intra_outline_similarities = {}

for outline_df in normalized_outlines:
    outline_name = outline_df['Outline'].iloc[0]
    vectors = outline_df.drop(columns=['Outline', 'Model']).values
    similarity_matrix = cosine_similarity(vectors)
    # 计算平均组内相似度（忽略对角线，即自身相似度）
    avg_intra_similarity = (np.sum(similarity_matrix) - np.trace(similarity_matrix)) / (similarity_matrix.shape[0] ** 2 - similarity_matrix.shape[0])
    intra_outline_similarities[outline_name] = avg_intra_similarity

# 2. 计算大纲之间的向量相似度
# 首先计算每个大纲的平均向量
outline_averages = {}

for outline_df in normalized_outlines:
    outline_name = outline_df['Outline'].iloc[0]
    vectors = outline_df.drop(columns=['Outline', 'Model']).values
    outline_average = np.mean(vectors, axis=0)
    outline_averages[outline_name] = outline_average

# cosine similarity
outline_names = list(outline_averages.keys())
outline_vectors = np.array(list(outline_averages.values()))
inter_outline_similarity_matrix = cosine_similarity(outline_vectors)

# 将大纲之间的相似度矩阵转换为 DataFrame 以便查看
inter_outline_similarity_df = pd.DataFrame(inter_outline_similarity_matrix, index=outline_names, columns=outline_names)

# 输出组内和大纲之间的相似度
print("cosine similarity：")
for outline_name, similarity in intra_outline_similarities.items():
    print(f"{outline_name}: {similarity:.4f}")

print("\n outline_averages：")
print(inter_outline_similarity_df)

# Concatenate all outlines into a single DataFrame for plotting
all_normalized_data = pd.concat(normalized_outlines, ignore_index=True)

# Dimensionality Reduction using PCA to reduce to 2D
features = all_normalized_data.drop(columns=['Outline', 'Model'])
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(features)

# Prepare data for visualization
plot_data = pd.DataFrame(reduced_data, columns=['Dim1', 'Dim2'])
plot_data['Outline'] = all_normalized_data['Outline']
plot_data['Model'] = all_normalized_data['Model']

# Plot the results with clusters for each outline
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Dim1', y='Dim2', hue='Outline', data=plot_data, palette='tab10', s=60, alpha=0.7)
plt.title("Dimensionality Reduction of Scores for Each Outline")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Outline", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)
plt.tight_layout()
plt.show()