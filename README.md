# プロジェクト概要

## 概要

情報系分野を専攻している北大大学院生3人による初めての共同開発プロジェクト．Kaggleのコンペティションに参加して，チーム開発の経験を積む．

## 開発者

- [TogoTarogit](https://github.com/TogoTarogit)
- [923ta](https://github.com/923ta)
- [ryosuke508](https://github.com/ryosuke508)

## プロジェクトの目的

- 共同開発の難易度を把握する
- 実社会応用と研究スキルのギャップを理解する

## 機械学習コンペについて

- プラットフォーム: Kaggle
- コンペティション: CommonLit
- 目的: 3-12年生の学生が書いた要約の品質を評価するモデルの精度を競う

## ソースコードの概要

- `setting.py`: 環境設定
- `train_T5.py`: T5モデルの訓練
- `calc_sim.py`: 類似度計算
- `requirement.txt`: 依存関係

## プロジェクトの過程

1. GitHubとDockerによる環境構築
2. 議事録をGoogle DocからNotionに移行
3. NLPとT5の調査
4. ポモドーロタイマーの導入
5. 次回のプロジェクトに向けて他のメンバーを招待

## 機械学習モデルの開発の過程
1. ## 使用技術とツール

> - Python, Jupyter Notebook
> - GitHub
> - Docker
> - Notion, Google Document
> - ポモドーロタイマー

2. ## 自然言語処理（Tokenizer）

> - T5
> - BERT

3. ## WordingとContentsの相関

> - 相関係数: 0.7513804859701966（wordingとcontents）
> - ![scatter](https://github.com/ProjectBarrele/CommonLit/assets/62383281/cb21f905-fffc-40e7-88f7-8be30acd49ba)

> - ランダム変数の相関係数: 0.051903052860149165
> - ![scatter_randam](https://github.com/ProjectBarrele/CommonLit/assets/62383281/1dc7aaa6-f7c2-418c-a724-d84bb7af004f)
> - 以上の調査により、wording の結果を利用することで、contents のある程度の精度を出せるのではないかと考えた。


4. ## モデルの仮決定

> - よりよい要約を出すようなモデルを作成し、そのモデルの出力との差をcontentsの点数とするしようと考えてた。
> - 要約生成モデル: T5
![content](https://github.com/ProjectBarrele/CommonLit/assets/76891064/77c62335-ea63-445f-82cb-5ea283e9bb12)

> - content と類似度（AIの作成した予約文と生徒の要約　文）の分布図。相関係数0.45。
> - 類似度算出モデル: 生成された要約と生徒の要約の類似度を計算
> - ![correlation_prompt_ext](https://github.com/ProjectBarrele/CommonLit/assets/62383281/db62efb7-57b4-48d6-873e-e33105b1d362)
![wording](https://github.com/ProjectBarrele/CommonLit/assets/76891064/ed5b8b16-3441-4b06-be32-27a09cb69ef3)



## 自分が頑張った点と自由記述

> ### ryosuke508

> > #### 頑張った点

> > - 過去開催されたコンペのモデル調査
> > - gitとDockerの使用
> > - WSLとUbuntuの操作
> > - Notionの作成
> > - LSTMモデルの実装と理解
> > - RNNの理解

> > #### 自由記述

> > 言語処理が初めての挑戦であり，多くの難しさに直面したが，多くの学びがあり，これを今後に活かしていきたい．

> ### 923ta

> > #### 頑張った点

> > - contents score, wording scoreの基礎モデルのコーディング
> > - 睡眠削減（起きていた）

> > #### 自由記述

> > このプロジェクトは「0を1にする能力」を向上させる良い機会であった．情報収集の力も更に向上させたいと感じる．

> ### togo

> > #### 頑張った点

> > - プロジェクト全体の管理
> > - 専門が異なる3人の協力の場を作成
> > - wordingとcontentsの相関とモデル全体の設計

> > #### 自由記述

> > 自分自身の技術力と社会で必要な力とのギャップを感じた．また，チーム開発においてマネジメントの重要性を認識した．このような取り組みを継続して行う必要性を感じた．

