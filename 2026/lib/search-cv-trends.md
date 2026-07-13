# コンピュータービジョン (Computer Vision) 技術の歴史的変遷と最新動向に関する体系的サーベイ・レポート

> **調査対象:** コンピュータービジョン (Computer Vision: CV) 分野における歴史的パラダイムシフトと最新基盤モデル・応用技術  
> **調査日 (Survey Date):** 2026-07-09 (2026年07月09日)  
> **更新日:** 2026-07-09 (2026年07月09日)  
> **関連資料:** [`sources_and_references.md`](file://sources_and_references.md) / [`papers_pdf/`](file://papers_pdf)

---

## エグゼクティブ・サマリー (Executive Summary)

コンピュータービジョン (CV) は、画像や動画データから意味的・幾何学的情報を抽出・理解・生成する技術領域である。2012年の ImageNet 競技会における深層学習 (CNN) の台頭以来、CV 分野の技術パラダイムは劇的な変遷を遂げてきた。

本サーベイ・レポートでは、2026年07月09日時点の最新知見に基づき、以下の主要なパラダイムシフトと技術動向を体系的に詳述する：

1. **CNN時代の歴史的発展:** 特徴量エンジニアリングから端から端まで (End-to-End) の深層特徴表現学習への転換と、深層畳み込みネットワーク（AlexNet, VGG, ResNet, EfficientNet等）の進化。
2. **Vision Transformer (ViT) による大域的アテンション革命:** 空間帰納バイアスからの脱却と、パッチ化自己アテンションおよび階層的ウィンドウ処理（Swin Transformer等）による汎用バックボーン化。
3. **自己教師あり学習 (Self-Supervised Learning - SSL):** 対照学習 (SimCLR, MoCo)、マスク画像モデリング (MAE)、および自律的知識蒸留 (DINO, DINOv2) を通じたラベルなし大規模事前学習の確立。
4. **視覚言語モデル (VLM) とマルチモーダルアライメント:** テキスト監督によるゼロショット認識 (CLIP, SigLIP) から、視覚指示チューニングに基づくオープン・商用マルチモーダル基盤モデル (LLaVA, InternVL, GPT-4o等) への進化。
5. **ゼロショット・プロンプタブルセグメンテーション:** 任意プロンプトに基づく汎用領域分割 (SAM) およびストリーミングメモリ機構による動画追跡への拡張 (SAM 2)。
6. **生成ビジョンと空間知能 (Spatial Intelligence / 3D・4D):** 潜在拡散モデル (SDXL, Flux) と動画生成モデル (Sora, DiT)、および実時間3Dシーン表現を革新した **3D Gaussian Splatting (3DGS)**。
7. **応用ドメインと今後の技術課題:** 自動運転・ロボティクス、医療・デジタル病理、産業外観検査の実装動向と、ハルシネーション・計算負荷・倫理的課題に対する処方箋。

---

## 目次 (Table of Contents)

1. [コンピュータービジョン技術の歴史的発展とパラダイムシフト](#1-コンピュータービジョン技術の歴史的発展とパラダイムシフト)  
   1.1 [CNN時代 (Convolutional Neural Networks) の勃興と進化](#11-cnn時代-convolutional-neural-networks-の勃興と進化)  
   1.2 [Vision Transformer (ViT) の登場と自己アテンションパラダイム](#12-vision-transformer-vit-の登場と自己アテンションパラダイム)  
   1.3 [自己教師あり学習 (Self-Supervised Learning - SSL) の進化](#13-自己教師あり学習-self-supervised-learning---ssl-の進化)  
2. [最新の動向・代表的基盤モデル (Vision Foundation Models & Multimodal)](#2-最新の動向代表的基盤モデル-vision-foundation-models--multimodal)  
   2.1 [視覚言語モデル (VLM / Multimodal Alignment) の発展](#21-視覚言語モデル-vlm--multimodal-alignment-の発展)  
   2.2 [ゼロショット・プロンプタブルセグメンテーション (SAM / SAM 2)](#22-ゼロショットプロンプタブルセグメンテーション-sam--sam-2)  
   2.3 [生成ビジョンと空間知能 (Spatial Intelligence / 3D・4D)](#23-生成ビジョンと空間知能-spatial-intelligence--3d4d)  
3. [歴代代表モデルの総合定量比較分析 (Comprehensive Comparative Analysis)](#3-歴代代表モデルの総合定量比較分析-comprehensive-comparative-analysis)  
4. [応用ドメインの最前線 (Domain Applications)](#4-応用ドメインの最前線-domain-applications)  
   4.1 [自動運転およびロボティクス (Autonomous Driving & Robotics)](#41-自動運転およびロボティクス-autonomous-driving--robotics)  
   4.2 [医療ビジョン・デジタル病理解析 (Medical & Pathology Vision)](#42-医療ビジョンデジタル病理解析-medical--pathology-vision)  
   4.3 [産業外観検査・異常検知・エッジ推論 (Industrial Inspection & Edge AI)](#43-産業外観検査異常検知エッジ推論-industrial-inspection--edge-ai)  
5. [技術的・社会的課題と今後の展望 (Open Challenges & Future Directions)](#5-技術的社会的課題と今後の展望-open-challenges--future-directions)  
6. [参考文献および原著論文カタログ導線](#6-参考文献および原著論文カタログ導線)

---

## 1. コンピュータービジョン技術の歴史的発展とパラダイムシフト

CV領域の進化は、アーキテクチャの変更にとどまらず、「特徴をどう定義・抽出するか」「目的関数と事前学習データは何か」という根源的な問いに対するパラダイムシフトの連続であった。

```
[古典的CV: Hand-crafted Features (SIFT/HOG)]
                    │ (2012 ImageNet Breakthrough)
                    ▼
[第一世代: CNN時代 (AlexNet -> VGG -> ResNet -> EfficientNet)]
                    │ (2020-2021 Spatial Inductive Biasからの脱却)
                    ▼
[第二世代: Vision Transformer (ViT -> Swin Transformer)]
                    │ (ラベル付きデータへの依存打破: SSL & Language Supervision)
                    ▼
[第三世代: 視覚基盤モデル (SSL: DINOv2/MAE / VLM: CLIP/SigLIP)]
                    │ (汎用プロンプタブル実行 & 3D空間知能への統合)
                    ▼
[第四世代: 空間知能 & 統合マルチモーダル基盤 (SAM 2 / 3DGS / GPT-4o / Sora)]
```

---

### 1.1 CNN時代 (Convolutional Neural Networks) の勃興と進化

深層畳み込みニューラルネットワーク (CNN) は、局所受容野 (Local Receptive Field)、重み共有 (Weight Sharing)、並進変異に対する不変性 (Translation Invariance) という視覚特性に適した空間帰納バイアス (Spatial Inductive Bias) を備える。

#### (1) AlexNet (2012) [[P01]](file://sources_and_references.md#L14)
- **時代背景と契機:** 人間が設計した特徴量（SIFT, HOG）とSVMの組み合わせが限界を迎える中、ILSVRC-2012 においてTop-5エラー率 15.3% を記録し、2位（従来手法、26.2%）に圧倒的差をつけて深層学習ブームの火付け役となった。
- **アーキテクチャと目的関数:**
  - 5層の畳み込み層と3層の全結合層で構成。
  - 活性化関数に非飽和関数 **ReLU ($f(x) = \max(0, x)$)** を採用し、勾配消失を和らげ学習を数倍高速化。
  - 過学習抑制のため **Dropout** とデータ拡張 (Data Augmentation) を導入。GPU並列処理用の分割アーキテクチャ。
  - 目的関数: 多クラス分類のクロスエントロピー損失。

#### (2) VGG / VGGNet (2014-2015) [[P02]](file://sources_and_references.md#L15)
- **時代背景と契機:** ネットワークの「深さ (Depth)」が視覚表現力に及ぼす影響を体系的に調査。
- **アーキテクチャ革新:**
  - 大きなフィルタサイズ（5x5や7x7）を排し、**小さな $3 \times 3$ 畳み込みフィルタ**のみをスタック。
  - $3 \times 3$ 畳み込みを2層重ねると実効的な受容野は $5 \times 5$、3層重ねると $7 \times 7$ に等価となる一方、パラメータ数が削減され、かつ非線形性（ReLU層の数）が増加して表現力が向上することを示した。
  - 代表例である VGG-16 / VGG-19 は、転移学習の標準的なバックボーンとして長年利用された。

#### (3) ResNet (Deep Residual Learning, 2016) [[P03]](file://sources_and_references.md#L16)
- **時代背景と契機:** 20層を超えてネットワークを深くすると、過剰学習ではなく最適化困難さによって訓練エラー率が悪化する「退行問題 (Degradation Problem)」が発生した。
- **アーキテクチャ革新と数理的根拠:**
  - **残差接続 (Residual / Shortcut Connection)** の導入。層の出力を $H(x)$ とする代わりに、残差 $F(x) = H(x) - x$ を学習させ、出力を $F(x) + x$ とする構成に変更した。
  
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x}$$
  
  - これにより恒等写像が容易に学習可能となり、誤差逆伝播法において勾配がショートカット経路を通じて入力側へ直接流れるため、100層・1,000層規模の深層学習が可能となった。
  - 各畳み込み層の直後に **Batch Normalization (BatchNorm)** を配置し、内部共変量シフトを解消。
  - 1x1 $\to$ 3x3 $\to$ 1x1 のボトルネック構造により計算効率を高め、ResNet-50 / ResNet-101 がILSVRC-2015分類・検出タスクで全圧勝した。

#### (4) EfficientNet & EfficientNetV2 (2019-2021) [[P04]](file://sources_and_references.md#L17), [[P05]](file://sources_and_references.md#L18)
- **時代背景と契機:** 深さ (Depth)、チャネル幅 (Width)、入力解像度 (Resolution) を場当たり的に手動拡張する従来手法に対し、最適計算量で最高性能を得る理論的スケーリング則を確立。
- **複合スケーリング則 (Compound Scaling):**
  - 複合係数 $\phi$ を用いて、ネットワークの深さ $d$、幅 $w$、解像度 $r$ を一定の制約式のもとで同時にスケーリングする：
  
  $$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$
  
  $$\text{s.t. } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$
  
- **特徴と進化:**
  - Mobile Inverted Bottleneck (MBConv) と Squeeze-and-Excitation (SE) ブロックをベースに、Neural Architecture Search (NAS) で設計。
  - **EfficientNetV2** では、初期層の MBConv を Fused-MBConv（1x1拡張と3x3畳み込みの統合）に置き換え、画像解像度とデータ拡張強さを連動させて上げる **Progressive Learning** を導入。パラメータ効率と学習速度を劇的に改善した。

---

### 1.2 Vision Transformer (ViT) の登場と自己アテンションパラダイム

2020年秋に発表された **Vision Transformer (ViT)** は、自然言語処理 (NLP) で標準となった Transformer Encoder を画像認識へ直接適用し、CNNが持っていた強い空間帰納バイアスからの脱却を果たした。

#### (1) Vision Transformer (ViT) (2021) [[P06]](file://sources_and_references.md#L22)
- **アーキテクチャ設計:**
  1. 入力画像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ をサイズ $P \times P$ の非重複パッチ $N = HW/P^2$ 個に分割し、1次元ベクトルに平坦化する。
  2. 線形投影により埋め込み次元 $D$ へ写像し、学習可能な1次元位置埋め込み (Positional Embedding) を加算する。
  3. 分類用トークン (`[CLS]`) を先頭に付与し、標準的な Transformer Encoder（マルチヘッド自己アテンション MSA と MLP）を適用。
- **計算複雑性と帰納バイアスの得失:**
  - 自己アテンションにより、最初の層から画像全体の大域的関係性を捕捉できる。
  - 一方で平並進不変性等の帰納バイアスを持たないため、中規模データ（ImageNet-1K単体）では過学習しやすいが、大規模事前学習（JFT-300MやImageNet-21K）を行うとCNNの精度を明確に凌駕するスケーリング特性を示す。

#### (2) DeiT (Data-efficient Image Transformers) (2021) [[P07]](file://sources_and_references.md#L23)
- **課題解消:** 外部の大規模データセットなしで ViT を ImageNet-1K 単体で高精度に学習させる枠組み。
- **ハード知識蒸留 (Hard-label Distillation):**
  - Transformer入力配列に `[DISTILL]` トークンを追加し、強力な CNN 教師モデル（RegNetY等）のハードラベル（ワンホット予測）に適合させる損失関数を導入。自己アテンションにCNNの局所構造バイアスを効果的に注入した。

#### (3) Swin Transformer & Swin V2 (2021-2022) [[P08]](file://sources_and_references.md#L24), [[P09]](file://sources_and_references.md#L25)
- **課題設定:** ViT の自己アテンション計算量は画像サイズ（トークン数 $N$）に対し $O(N^2)$ で増大するため、高解像度画像や物体検出・セグメンテーション等の密予測タスクへ適用が困難であった。
- **アーキテクチャ革新:**
  - **Shifted Window Attention (W-MSA / SW-MSA):** 画像をローカルなウィンドウに分割し、ウィンドウ内部のみで自己アテンションを計算 ($O(N)$ 線形計算量)。連続する層間でウィンドウ位置をシフトさせることで、ウィンドウをまたぐ情報伝達を保証した。
  - **Patch Merging による階層化:** 解像度を段階的に下げながらチャネル数を増やし、CNNのFeature Pyramidに似た多解像度表現を構築。これにより COCO 検出や ADE20K セグメンテーションにおけるデファクトスタンダード・バックボーンとなった。

---

### 1.3 自己教師あり学習 (Self-Supervised Learning - SSL) の進化

手動ラベル付けのコストとデータ量の限界を突破するため、ラベルなし画像からの自己教師あり表現学習 (SSL) が発展した。

```
[SSLの3大アプローチ]
 ├── 対照学習 (Contrastive Learning): SimCLR, MoCo
 │     └─ 同一画像の異なる拡張ビューの類似度最大化 / 異画像との対比 (InfoNCE損失)
 ├── 非対照・自己蒸留 (Non-Contrastive / Self-Distillation): BYOL, DINO, DINOv2
 │     └─ 負例対不要、Student/Teacherの出力一致や正規化特徴合わせ
 └── マスク画像モデリング (Masked Image Modeling - MIM): MAE
       └─ パッチの大部分(75%)をマスクし、可視パッチのみから画素空間・特徴を再構成
```

#### (1) SimCLR & MoCo (Contrastive Learning, 2020) [[P10]](file://sources_and_references.md#L29), [[P11]](file://sources_and_references.md#L30)
- **SimCLR:** 同一画像に対して2種類の強力なデータ拡張（ランダムクロップ、色調変換等）を施して正例対とし、バッチ内の他画像ビューを負例対として **InfoNCE (NT-Xent) 損失** を最小化する。超大バッチサイズ（4096等）と非線形射影ヘッドにより線形プロービング精度が大幅に向上。
- **MoCo:** 大バッチサイズへの依存を解消するため、過去の負例表現を格納する動的辞書キューと、パラメータを移動平均で更新するモメンタムエンコーダ (Momentum Encoder) を設計。

#### (2) Masked Autoencoders (MAE) (2022) [[P13]](file://sources_and_references.md#L32)
- **アーキテクチャと目的関数:**
  - NLPの BERT に類似した Masked Image Modeling (MIM)。
  - 画像パッチの **75%** という非常に高い割合をランダムにマスクする。
  - **非対称 Encoder-Decoder:** 重い ViT Encoder は「マスクされていない25%の可視パッチのみ」を処理するため、計算効率が極めて高い。軽量な Decoder がマスク位置を含む全トークンを受け取り、画素値の平均二乗誤差 (MSE) で元パッチを再構成する。

#### (3) DINO & DINOv2 (2021-2024) [[P14]](file://sources_and_references.md#L33), [[P15]](file://sources_and_references.md#L34)
- **DINO (Self-distillation with no labels):** Student と Teacher ネットワーク間でのクロスエントロピー損失を最小化。崩壊を防ぐため Teacher 出力に Centering と Sharpening を適用。ViT のアテンションマップに物体の意味的領域が自律的に浮かび上がることが判明した。
- **DINOv2 (汎用視覚表現基盤モデル):**
  - **精選データプール LVD-142M:** ウェブからの無差別収集ではなく、精選された高品質・多様な 1億4,200万枚 画像データセットを作成。
  - 画像レベルの DINO 損失とパッチレベルの iBOT 損失（マスクパッチ予測）を統合学習。
  - **転移性能:** ファインチューニングなしで凍結特徴量のみを用いた k-NN 分類や線形プロービング、深度推定・セグメンテーションで最高の汎化能力を実現した。

---

## 2. 最新の動向・代表的基盤モデル (Vision Foundation Models & Multimodal)

### 2.1 視覚言語モデル (VLM / Multimodal Alignment) の発展

視覚特徴と言語特徴を共通の埋め込み空間へアラインメントするアプローチは、ゼロショット画像認識とマルチモーダル会話AIを可能にした。

#### (1) CLIP (Contrastive Language-Image Pre-training, 2021) [[P16]](file://sources_and_references.md#L38)
- **対照アライメント設計:**
  - Image Encoder（ViT または ResNet）と Text Encoder（Transformer）によるデュアルエンコーダ構成。
  - 4億対のウェブ画像テキストペア (WIT-400M) に対し、バッチ内の $N$ 個の正例ペアのコサイン類似度を最大化し、$N^2 - N$ 個の負例ペアの類似度を最小化する対称型 InfoNCE 損失を適用。
  - 任意の分類タスクにおいて、クラス名をテキストプロンプトとしてゼロショットで分類可能となり、分布外データに対する極めて高い頑健性を実証した。

#### (2) SigLIP (Sigmoid Loss for Language Image Pre-Training, 2023) [[P17]](file://sources_and_references.md#L39)
- **Sigmoid損失によるバッチ正規化からの解放:**
  - CLIPの Softmax 対照損失はバッチ全体に対する正規化計算を伴うため通信コストが高い。
  - SigLIP は画像・テキストペアの組み合わせ各々を独立した二値分類（マッチするか否か）の **シグモイド交差エントロピー損失** で処理する。
  
  $$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{B^2} \sum_{i=1}^B \sum_{j=1}^B \log \frac{1}{1 + \exp(-z_{ij} \cdot (-1)^{\mathbb{I}_{i \neq j}})}$$
  
  - メモリ効率が大きく改善され、より大きなバッチ・より良い表現アライメントを獲得した。

#### (3) BLIP-2 & LLaVA シリーズ (オープンソースマルチモーダルLLM) [[P18]](file://sources_and_references.md#L40), [[P19]](file://sources_and_references.md#L41)
- **BLIP-2:** 事前学習済みの強力な視覚エンコーダ（EVA-CLIP等）と凍結した LLM を軽量な **Q-Former (Querying Transformer)** で接続。少ない学習パラメータで視覚質問応答 (VQA) の最高精度を実現。
- **LLaVA (Visual Instruction Tuning):**
  - CLIP/SigLIP の視覚エンコーダと LLM (Llama, Vicuna等) を線形変換や MLP アダプタで連結し、GPT-4 を用いて合成した視覚指示チューニングデータセットで自己対話的に学習。
  - **InternVL 2 [[P20]](file://sources_and_references.md#L42):** 入力画像の解像度に応じて動的にタイル分割を行う Dynamic High-Resolution 構成と、自社開発の InternViT-6B エンコーダによって、オープンソースながら GPT-4V や Gemini Pro に匹敵する視覚推論能力を発揮する。

---

### 2.2 ゼロショット・プロンプタブルセグメンテーション (SAM / SAM 2)

画像中のあらゆる対象を、学習時のカテゴリに依存せずゼロショットで分割する技術は、CVのインタラクティブ操作に革命をもたらした。

```
[SAM 2: Static Image & Streaming Video Segmentation Architecture]
  画像/動画フレーム ──> [Hiera ViT Image Encoder] ──> フレーム特徴量
                                                              │
  ユーザープロンプト ──> [Prompt Encoder] ──────────────────────┼──> [Mask Decoder] ──> マスク予測 & 確信度
  (点・枠・マスク)                                            │
                                                              ▼
  過去フレーム情報   ──> [Memory Bank] <── [Memory Attention (Cross-Attn)]
```

#### (1) SAM (Segment Anything Model, 2023) [[P21]](file://sources_and_references.md#L46)
- **3大構成要素:**
  1. **Image Encoder:** MAE 事前学習済みの ViT-H/16。高解像度画像の空間埋め込みを一度だけ計算。
  2. **Prompt Encoder:** 点・ボックス・マスク・テキスト等のプロンプトを位置エンコーディングで柔軟に表現。
  3. **Mask Decoder:** 画像特徴とプロンプト特徴の交差アテンション計算によって、リアルタイム (ブラウザ上で約50ms) に分割マスクおよび曖昧性対応の複数候補・確信度 (IoU) を出力する。
- **データエンジンと SA-1B:** モデルとアノテーターが協調するデータ構築ループにより、1,100万枚の画像に11億個の高品質マスクを付与した史上最大のセグメンテーションデータセット SA-1B を公開。

#### (2) SAM 2 (Segment Anything Model 2 for Images and Videos, 2024年8月) [[P22]](file://sources_and_references.md#L47)
- **動画・時系列空間への統一拡張:**
  - SAM のタスク設定を動画へと拡張。各フレームの対象オブジェクトのマスクを追跡・分割する。
- **ストリーミングメモリ機構 (Streaming Memory Architecture):**
  - **Memory Bank:** 過去フレームの予測結果や特徴量（空間特徴メモリおよびオブジェクトポインタ）を保持。
  - **Memory Attention:** 現在のフレーム処理時に Cross-Attention で過去メモリを参照し、フレーム間での物体の形状変化や一時的遮蔽 (Occlusion) に強固に対応する。
- **性能対比:** 従来の動画オブジェクトセグメンテーション (VOS) モデルより **3倍少ないインタラクション数** で高精度な追跡を実現。さらに静止画タスクにおいても、階層的バックボーン Hiera の採用により初代 SAM 比で **6倍高速** かつ優れた精度を達成した。

---

### 2.3 生成ビジョンと空間知能 (Spatial Intelligence / 3D・4D)

画像・動画認識にとどまらず、物理世界の3D構造やダイナミクスを理解・生成する技術が急速に発展している。

#### (1) 拡散モデルと動画生成世界モデル (Sora, DiT) [[P23]](file://sources_and_references.md#L51), [[P24]](file://sources_and_references.md#L52)
- **Latent Diffusion Models (LDM):** ピクセル空間ではなく VAE の圧縮潜在空間でノイズ除去を行うことで生成の高解像度化と効率化を達成 (Stable Diffusion / SDXL / Flux.1)。
- **Diffusion Transformer (DiT):** U-Net バックボーンを Transformer に入れ替えるスケーリング技術。動画の時空間パッチ表現を扱う Sora 等においては、単なる動画生成にとどまらない「物理世界のシミュレータとしての空間知能 (World Model)」の可能性を示している。

#### (2) 3D Gaussian Splatting (3DGS, SIGGRAPH 2023) [[P25]](file://sources_and_references.md#L53)
- **技術課題と変革:** NeRF (Neural Radiance Fields) が暗黙的な関数 representation とボリュームレンダリングにより膨大なクエリ評価時間を要したのに対し、**3DGS は明示的な異方性3Dガウシアン集合による表現と可視性考慮ラスタライズ**によって、実時間での極めて高品質な新規視点画像生成を確立した。
- **数理的 formulation:**
  - シーンは数百万個の 3D ガウシアンで表現される。個々のガウシアンは中心位置（平均ベクトル）$\boldsymbol{\mu} \in \mathbb{R}^3$ と共分散行列 $\boldsymbol{\Sigma}$ を有する。
  
  $$G(\mathbf{x}) = \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$
  
  - 最適化可能パラメータとして、各ガウシアンは $\boldsymbol{\mu}$, 共分散を構成するクォータニオン回転 $\mathbf{r}$ とスケール $\mathbf{s}$, 不透明度 $\alpha \in [0, 1]$, および視線方向に依存する色を表現する球面調和関数 (Spherical Harmonics: SH) 係数を持つ。
- **高速タイルラスタライザと適応的密度制御:**
  - スクリーン空間を $16 \times 16$ ピクセルのタイルに分割し、視錐台カリングと高速基数ソートによってアルファブレンディングを実行する。
  - 最適化中に誤差が大きい領域ではガウシアンを複製・分割する **Adaptive Density Control** を導入。
  - **性能:** Mip-NeRF 360 データセット等において、NeRF と同等以上のレンダリング忠実度 (PSNR ~30 dB) を維持しながら、**1080p 解像度で 100 FPS 以上** という驚異的なレンダリング速度を実現した。

---

## 3. 歴代代表モデルの総合定量比較分析 (Comprehensive Comparative Analysis)

主要な時代のブレークスルー・モデルについて、バックボーン設計、事前学習規模、計算要件、および代表的ベンチマーク性能を統合的に比較する。

| モデル名 / 時代 | 発表年 | 代表バックボーン・パラメータ数 | 事前学習データセット・規模 | 画像分類 (ImageNet Top-1) | 密予測 / タスク別代表精度 | 公開ライセンス | 主なブレークスルーポイント |
|---|---|---|---|---|---|---|---|
| **AlexNet** | 2012 | 8 layers (60M params) | ILSVRC-2012 (1.2M images, ラベル付) | 63.3% (Top-5 Error 15.3%) | — | BSD | GPU並列実装・ReLU・Dropoutによる深層学習の原点 |
| **ResNet-152** | 2016 | 152 layers (60M params) | ImageNet-1K (1.2M images) | ~78.3% (後年レシピで ~81.8%) | COCO Det: 40+ mAP | MIT | 残差接続による深層化と退行問題の完全克服 |
| **EfficientNet-B7** | 2019 | MBConv NAS (66M params) | ImageNet-1K / AutoAugment | **84.3%** | — | Apache-2.0 | 深さ・幅・解像度の理論的複合スケーリング則 |
| **ViT-H/14** | 2021 | Transformer (632M params) | JFT-300M (300M images, proprietary) | **88.55%** | — | Apache-2.0 | パッチトークン自己アテンションによるCNNからの移行 |
| **Swin-L** | 2021 | Hierarchical ViT (197M params) | ImageNet-22K (14M images) | **87.3%** | COCO Det: **58.7 mAP** / ADE20K: **53.5 mIoU** | MIT | Shifted Windowによる線形計算量と密予測SOTA |
| **MAE (ViT-H)** | 2022 | Asymmetric MIM (632M params) | ImageNet-1K (ラベルなし自己教師学習) | **87.8%** (ファインチューニング) | — | CC-BY-NC 4.0 | 高マスキング率(75%)自己再構成学習による強力な表現獲得 |
| **DINOv2-g** | 2024 | ViT-Giant (1.1B params) | LVD-142M (142M curated images) | **84.5%** (Linear / k-NN, 凍結特徴) | Monocular Depth & ADE20K zero-shot SOTA | Apache-2.0 | 142M精選データ自己学習によるタスク非依存汎用視覚表現 |
| **CLIP (ViT-L/14)** | 2021 | Dual Encoder (428M params) | WIT-400M (400M image-text pairs) | **76.2%** (Zero-shot / 未学習) | 強力な分布外汎化性 (ImageNet-R/A) | MIT | テキスト対照アライメントによるゼロショットオープン認識 |
| **SAM 2 (Hiera-L)**| 2024 | Hiera ViT + Memory Bank | SA-1B + SA-V (50.9K videos, 642K masks)| — | Video J&F Score SOTA / 初代SAM比 **6倍高速** | Apache-2.0 | ストリーミングメモリによる動画・画像統合ゼロショットセグメンテーション |
| **3DGS** | 2023 | 明示的異方性 3D Gaussian | Mip-NeRF 360 実写マルチビュー画像 | — | PSNR: **30.08 dB** / Rendering: **130+ FPS** | Inria Non-Com. | 異方性ガウシアンとタイルラスタライザによる実時間空間表現 |

---

## 4. 応用ドメインの最前線 (Domain Applications)

### 4.1 自動運転およびロボティクス (Autonomous Driving & Robotics)

- **Bird's-Eye-View (BEV) 統合知覚:** 車載の複数視点カメラ・LiDARから時空間 Transformer (BEVFormer [[P27]](file://sources_and_references.md#L57)) を通じて俯瞰鳥瞰図 (BEV) クエリへ特徴を統合し、3D物体検出・占有格子予測 (Occupancy Prediction) を実行。
- **End-to-End 自動運転モデル:** 知覚・軌道予測・行動計画を個別のパイプラインに分断せず、単一ネットワーク内でエンドツーエンドに学習する UniAD が進展。
- **Vision-Language-Action (VLA) モデル:** 視覚観測と言語指示からロボットの関節操作やエンドエフェクタ動作コマンドを直接自己回帰生成する RT-2 や OpenVLA が実世界ロボットの一般化を推進している。

### 4.2 医療ビジョン・デジタル病理解析 (Medical & Pathology Vision)

- **Whole Slide Image (WSI) 特有のスケール課題:** 病理スライドは数億～百億ピクセル規模 (ギガピクセル) に及ぶため、従来の小さくクロップした個別パッチ学習ではスライド全域の予後・微小病変組織連関を見落とす限界があった。
- **病理特化基盤モデルの革新:**
  - **UNI [[P28]](file://sources_and_references.md#L58):** 1億枚を超える病理パッチで事前学習された汎用視覚表現モデル。
  - **Prov-GigaPath [[P29]](file://sources_and_references.md#L59) (Nature 2024):** LongNet アーキテクチャを適用し、1.3億タイル・17万枚超の WSI を対象に、スライド全域の超長配列トークン間関係をモデリング。がん亜型分類や病理ゲノム解析ベンチマークで従来法を大きく打破した。

### 4.3 産業外観検査・異常検知・エッジ推論 (Industrial Inspection & Edge AI)

- **正常データのみに基づく異常検知:** 工業現場では異常画像の網羅的収集が不可能なため、正常品の ImageNet 学習済特徴量をメモリバンク（コアセット）に保存して特徴間距離を計測する **PatchCore [[P30]](file://sources_and_references.md#L60)** が高速・超高精度な検査標準を構築。
- **エッジ実装とモデル量子化:** 工場の生産ラインやモバイル端末でのリアルタイム推論に向け、INT8 / FP8 への実行時量子化 (TensorRT, ONNX Runtime) や、CNN と Attention をハイブリッド構成する MobileViT / FastViT 等の軽量推論技術が成熟している。

---

## 5. 技術的・社会的課題と今後の展望 (Open Challenges & Future Directions)

1. **視覚言語モデルにおけるハルシネーションと空間的グラウンディングの改善**
   - VLM はテキストの言語的な共起確率に引っ張られ、画像に実在しない物体を描写するハルシネーション現象が発生しやすい。物体座標 (Bounding Box/Mask) の明示的グラウンディング学習による改善が急務である。
2. **長時間の高解像度動画・4D処理における計算効率と環境負荷**
   - 4K/8K 動画や 3D/4D ガウシアンのリアルタイム追跡・生成モデルは、GPU VRAM 消費量および推論電力が甚大となる。状態間スパースアテンションや軽量表現の確立が求められる。
3. **データセットの著作権・倫理・プライバシー問題**
   - 大規模学習データの収集手法に対する透明性要請と、ライセンスクリーンなデータプール構築 (SA-1B, LVD-142M 等) の重要性が高まっている。

---

## 6. 参考文献および原著論文カタログ導線

本レポートにおいて解説・参照したすべての原著論文、DOI / arXiv リンク、および公式 GitHub リポジトリ等の完全な出典カタログは、リポジトリ内の独立資料として整理されている：

- **文献およびソースカタログ:** [`sources_and_references.md`](file://sources_and_references.md)
- **論文PDFコレクション構成:** [`papers_pdf/README.md`](file://papers_pdf/README.md)
