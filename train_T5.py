# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

""" main """
from pathlib import Path
import re
import math
import time
import copy
from tqdm import tqdm
import pandas as pd
import tarfile
import neologdn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys
# sys.path.append('../input/settings/')
# import settings

os.makedirs('./data/commonlit-evaluate-student-summaries', exist_ok=True)
data_dir_path = Path('./data/commonlit-evaluate-student-summaries')

body_data = pd.read_csv(data_dir_path.joinpath('prompts_train.csv'))
summary_data = pd.read_csv(data_dir_path.joinpath('summaries_train.csv'))

def join_text(x, add_char='。'):
    return add_char.join(x)

# トークナイズ化前のテキストの正規化
def preprocess_text(text):
    text = re.sub(r'[\r\t\n\u3000]', '', text)
    text = neologdn.normalize(text)
    text = text.lower()
    text = text.strip()
    return text

# 本文と要約を結合
data = pd.merge(
    # マージするためにカラム名を変更
    body_data.rename(columns={'prompt_text': 'body_text'}),
    summary_data.rename(columns={'text': 'summary_text'}),
    on='prompt_id', how='inner'
).assign(
    body_text=lambda x: x.body_text.map(lambda y: preprocess_text(y)),
    summary_text=lambda x: x.summary_text.map(lambda y: preprocess_text(y))
)

def convert_batch_data(train_data, valid_data, tokenizer):

    def generate_batch(data):

        batch_src, batch_tgt = [], []
        for src, tgt in data:
            batch_src.append(src)
            batch_tgt.append(tgt)

        batch_src = tokenizer(
            batch_src, max_length=settings.max_length_src, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_tgt = tokenizer(
            batch_tgt, max_length=settings.max_length_target, truncation=True, padding="max_length", return_tensors="pt"
        )

        return batch_src, batch_tgt

    train_iter = DataLoader(train_data, batch_size=settings.batch_size_train, shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(valid_data, batch_size=settings.batch_size_valid, shuffle=True, collate_fn=generate_batch)

    return train_iter, valid_iter

tokenizer = T5Tokenizer.from_pretrained(settings.MODEL_NAME, is_fast=True)

X_train, X_test, y_train, y_test = train_test_split(
    data['body_text'], data['summary_text'], test_size=0.2, random_state=42, shuffle=True
)

train_data = [(src, tgt) for src, tgt in zip(X_train, y_train)]
valid_data = [(src, tgt) for src, tgt in zip(X_test, y_test)]

train_iter, valid_iter = convert_batch_data(train_data, valid_data, tokenizer)

class T5FineTuner(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(settings.MODEL_NAME)

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None,
        decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
def train(model, data, optimizer, PAD_IDX):
    
    model.train()
    
    loop = 1
    losses = 0
    pbar = tqdm(data)
    for src, tgt in pbar:
                
        optimizer.zero_grad()
        
        labels = tgt['input_ids'].to(settings.device)
        labels[labels[:, :] == PAD_IDX] = -100

        outputs = model(
            input_ids=src['input_ids'].to(settings.device),
            attention_mask=src['attention_mask'].to(settings.device),
            decoder_attention_mask=tgt['attention_mask'].to(settings.device),
            labels=labels
        )
        loss = outputs['loss']

        loss.backward()
        optimizer.step()
        losses += loss.item()
        
        pbar.set_postfix(loss=losses / loop)
        loop += 1
        
    return losses / len(data)

def evaluate(model, data, PAD_IDX):
    
    model.eval()
    losses = 0
    with torch.no_grad():
        for src, tgt in data:

            labels = tgt['input_ids'].to(settings.device)
            labels[labels[:, :] == PAD_IDX] = -100

            outputs = model(
                input_ids=src['input_ids'].to(settings.device),
                attention_mask=src['attention_mask'].to(settings.device),
                decoder_attention_mask=tgt['attention_mask'].to(settings.device),
                labels=labels
            )
            loss = outputs['loss']
            losses += loss.item()
        
    return losses / len(data)

model = T5FineTuner()
model = model.to(settings.device)

optimizer = optim.Adam(model.parameters())

PAD_IDX = tokenizer.pad_token_id
best_loss = float('Inf')
best_model = None
counter = 1

for loop in range(1, settings.epochs + 1):

    start_time = time.time()

    loss_train = train(model=model, data=train_iter, optimizer=optimizer, PAD_IDX=PAD_IDX)

    elapsed_time = time.time() - start_time

    loss_valid = evaluate(model=model, data=valid_iter, PAD_IDX=PAD_IDX)

    print('[{}/{}] train loss: {:.4f}, valid loss: {:.4f} [{}{:.0f}s] counter: {} {}'.format(
        loop, settings.epochs, loss_train, loss_valid,
        str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
        elapsed_time % 60,
        counter,
        '**' if best_loss > loss_valid else ''
    ))

    if best_loss > loss_valid:
        best_loss = loss_valid
        best_model = copy.deepcopy(model)
        counter = 1
    else:
        if counter > settings.patience:
            break

        counter += 1
        
# 学習済みモデルの保存
model_dir_path = Path('model')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)
tokenizer.save_pretrained(model_dir_path)
best_model.model.save_pretrained(model_dir_path)

def generate_text_from_model(text, trained_model, tokenizer, num_return_sequences=1):

    trained_model.eval()
    
    text = preprocess_text(text)
    batch = tokenizer(
        [text], max_length=settings.max_length_src, truncation=True, padding="longest", return_tensors="pt"
    )

    # 生成処理を行う
    outputs = trained_model.generate(
        input_ids=batch['input_ids'].to(settings.device),
        attention_mask=batch['attention_mask'].to(settings.device),
        max_length=settings.max_length_target,
        repetition_penalty=8.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        # temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
        # num_beams=10,  # ビームサーチの探索幅
        # diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
        # num_beam_groups=10,  # ビームサーチのグループ
        num_return_sequences=num_return_sequences,  # 生成する文の数
    )

    generated_texts = [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in outputs
    ]

    return generated_texts

# インスタンス化
tokenizer = T5Tokenizer.from_pretrained(model_dir_path)
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir_path)
trained_model = trained_model.to(settings.device)
                                
index = 1
body = valid_data[index][0]
summaries = valid_data[index][1]
generated_texts = generate_text_from_model(
    text=body, trained_model=trained_model, tokenizer=tokenizer, num_return_sequences=1
)
print('□ 生成本文')
print('\n'.join(generated_texts[0].split('。')))
print()
print('□ 教師データ要約')
print('\n'.join(summaries.split('。')))
print()
print('□ 本文')
print(body)   

# todo
