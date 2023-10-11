
generated_text = """the three elements of an ideal tragedy as described by aristotle is that it should have a complex plot, imitate actions that excite pity and fear into the audience. 
"it follows plainly, in the first place, that the change of fortune presented must not be the spectacle of a virtuous man brought from prosperity to adversity: for this moves neither pity nor fear; it merely shocks us.'
"""


import os 
from pathlib import Path
import pandas as pd

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import matplotlib.pyplot as plt


data_dir_path = Path('./data/commonlit-evaluate-student-summaries')

body_data = pd.read_csv(data_dir_path.joinpath('prompts_train.csv'))
summary_data = pd.read_csv(data_dir_path.joinpath('summaries_train.csv'))
summary_data = summary_data[summary_data['prompt_id']=='39c16e']
summary_data = summary_data.reset_index(drop=True)

# print(summary_data.head())


print(body_data.columns)
print(summary_data.columns)
# Index(['prompt_id', 'prompt_question', 'prompt_title', 'prompt_text'], dtype='object')
# Index(['student_id', 'prompt_id', 'text', 'content', 'wording'], dtype='object')

# BERTのモデルとトークナイザのロード
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
def calc_similality(generated_txt ,target_txt):
    # 文章のトークン化
    inputs1 = tokenizer(generated_txt, return_tensors='pt', truncation=True, padding=True)
    inputs2 = tokenizer(target_txt, return_tensors='pt', truncation=True, padding=True)

    # モデルを通じて埋め込みの取得
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # コサイン類似度の計算
    cosine_sim = cosine_similarity(embeddings1, embeddings2)
    # ユークリッド距離の計算
    # euclidean_dist = distance.euclidean(embeddings1, embeddings2)
    return  cosine_sim[0][0]



# print(summary_data.head())
similarity_list = []
for index in range(len(summary_data)):
    target_txt = summary_data['text'][index]
    sim = calc_similality(generated_text, target_txt)
    similarity_list.append(sim)


sim_df = pd.DataFrame(similarity_list, columns=['similarity'])

# print(similarity_list)
# print(sim_df.shape)
summary_data = pd.concat([summary_data, sim_df], axis=1)

# print(summary_data.head())
summary_data.to_csv('./summaries_train_add_sim.csv', index=False)

# 'A'と'B'の2つの列を取り出す
x = summary_data['content']
y = summary_data['similarity']

# 相関図の表示
plt.scatter(x, y, color='blue', label='Data Points')
plt.title('Correlation between Column A and Column B')
plt.xlabel('content')
plt.ylabel('similarity')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('correlation.png')

correlation = x.corr(y)
print(correlation)
print("finish calc similarity")
'''
todo 
extraxt target Content 
'''
