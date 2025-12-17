# NKUST 類神經網路期末報告

隊伍：TEAM_8929  
隊員：王詩芸  

Private leaderboard：0.734497 / Rank 139  

---

## 壹、資料處理與程式環境

### 一、資料前處理

本競賽使用心臟電腦斷層（Cardiac CT）影像資料，共 50 位病人，每筆資料包含三維 CT 影像（NIfTI 格式）與對應之人工標註分割結果。

標註類別定義：

Class 0：背景（Background）  

Class 1：心肌（Myocardium）  

Class 2：主動脈瓣（Aortic Valve）  

Class 3：鈣化病灶（Calcification）  

由於各類別在影像中呈現的體積、空間分佈與出現頻率差異極大，尤其 class3 屬於極小病灶（tiny lesion），若直接以單一模型同時處理所有類別，容易造成模型偏向大型結構而忽略微小病灶。

每個標註的特徵:

Class 0(背景)：佔整體影像體積的絕大多數，對模型訓練而言資訊密度低。  

Class 1(心肌)：屬於連續、結構完整且體積相對穩定的區域。  

Class 2(主動脈瓣)：體積明顯小於心肌，結構較薄，且在不同切片中形態變化大。  

Class 3(鈣化)：體積最小，且高度稀疏，在多數病患中僅佔整體體積的極小比例，甚至低於 1%。並且病患間差異極大。（有無鈣化、數量、分布位置差異明顯）  

因此在資料處理與模型設計階段即採取分流式策略。

#### （一）影像標準化與格式統一

所有 CT 影像皆進行以下前處理步驟：

1.影像讀取並統一為單通道格式  
2.影像方向標準化為 RAS 座標系  
3.強度正規化：  

(1)HU範圍設定為  

(2)線性映射至區間  

#### （二）資料分流與類別重組

依照標註特性，將分割問題拆解為三個子任務：

1.Class1 + Class2 分割任務  

(1)專注於穩定且體積較大的解剖結構  

(2)訓練時忽略 class3、class0  

2.Class3 分割任務  

僅針對鈣化病灶(class3)進行二元分割，標註重新映射為0：非鈣化、1：鈣化。  

3.Class3 病人層級分類任務  

作為 segmentation pipeline 的輔助決策模組，判斷整個病人是否存在鈣化病灶。

---

### 二、原始程式碼概述

整體專案以 Python 為主，並使用 PyTorch 與 MONAI 框架實作。

主要程式模組如下：

- train_class12.py：訓練 class1（心肌）與 class2（主動脈瓣）的分割模型。  
- train_class3.py：針對 class3 鈣化病灶進行 tiny lesion segmentation 。  
- train_class3_classifier.py：病人層級是否存在鈣化的分類模型。  
- infer_and_dice.py：多模型推論與最終分割結果融合。  

---

### 三、執行環境

作業系統：Windows 10  

GPU：NVIDIA RTX 4080（16 GB VRAM）  

CPU：i7-13700K 架構  

RAM：64 GB  

Python：3.10  

深度學習套件：PyTorch 2.5.1、CUDA 12.1、MONAI 1.1.0、NumPy、Nibabel  

---

## 貳、模型設計與訓練策略

### 一、模型訓練原始程式碼與說明

#### (一) train_class12.py（Class1 + Class2 Segmentation）

目的：訓練心肌（class1）與主動脈瓣（class2）分割模型，忽略 class3，提升大結構的穩定收斂與泛化能力。

##### (1) 資料輸入與資料集建立

輸入：

training_image/patientXXXX.nii.gz  

training_label/patientXXXX_gt.nii.gz  

以固定比例切分 train/val（例如 40/10）。

使用 MONAI transforms 建立資料管線，常見步驟包含：

LoadImaged、EnsureChannelFirstd  

Orientationd(axcodes="RAS")  

ScaleIntensityRanged(a_min=-1000, a_max=2000, b_min=0, b_max=1)  

Patch-based sampling（依設定 ROI size 例如 96³ 或 128³）  

Label 處理：在訓練 class1+2 時，通常會將標註轉為「只關心 class1、class2」的形式（如 one-hot 或類別篩選），使 loss 不被 class3和 class0 干擾。

##### (2) 模型建立

使用 SwinUNETR(in_channels=1, out_channels=NUM_CLASSES) 建立分割網路。

out_channels 依設計為：3 類（background, class1, class2）

##### (3) Loss、Optimizer 與 AMP

Loss：Dice + Cross Entropy  

Optimizer：AdamW  

使用 AMP（autocast + GradScaler）加速訓練並降低 GPU 記憶體需求。

##### (4) 訓練迴圈與驗證

model.train() 計算 train loss  

model.eval() 在 validation set 計算 Dice  

指標輸出：輸出各類別 Dice 與平均 Dice，作為最佳模型保存依據。

##### (5) Early stopping 與權重保存

以 validation Dice（或平均 Dice）為主要 early stopping 指標。

當連續20個 epoch 無提升，停止訓練。

---

### 二、train_class3.py（Class3 Tiny Lesion Segmentation）

目的：針對鈣化（class3）建立專用分割模型。由於 class3 體積極小且僅少數病人出現，因此採用更聚焦的資料抽樣與評估策略。

##### (1) 資料掃描與正負樣本處理

程式會先掃描所有 label：統計有無 class3（你實際結果：Total 50；Positive 10；Negative 40）。

訓練集以 positive case 為主（或混入部分 negative case）。

val 也以 positive case 為主，確保評估指標能反映病灶偵測能力。

##### (2) Label 重映射（Binary segmentation）

將原多類別標註重映射為二元：label == 3 → 1、其他 → 0。

使模型專注學習「鈣化 vs 非鈣化」。

##### (3) 影像前處理與 patch 抽樣策略

前處理同樣包含 RAS 方向與強度正規化。

重要差異：class3 使用「強烈偏向病灶區域」的 patch 抽樣（例如 RandCropByLabelClassesd 對 class3 ratio 提高），以提高 tiny lesion 出現頻率，避免模型在訓練中看不到正樣本。

##### (4) 模型與 Loss

使用 SwinUNETR 獨立訓練二元輸出（out_channels=2）。

Loss 常採 Dice-based / Dice+CE 的變體，並讓訓練目標更偏向 Recall（避免漏檢）。

##### (5) Tiny lesion 評估指標與 early stopping

Validation 不以 Dice 為主（因為 tiny lesion 下 Dice 對位移極敏感），改用 Recall（敏感度）、Precision、Lesion-level F1（病灶層級是否偵測到）

早停與 best model 儲存依據：

Score = (Recall + LesionF1) / 2

score 提升即存檔，無提升累積 patience 直到 early stop。

你訓練紀錄中，Recall 有明顯上升並能存到 best model，代表偵測能力建立成功。

---

### 三、train_class3_classifier.py（Patient-level HasCalc Classifier）

目的：建立病人層級「是否存在 class3 鈣化」的二元分類器，作為 pipeline 的輔助模組，降低無鈣化病人出現假陽性的風險。

##### (1) 標籤定義與資料選取

由 GT label 判定：影像中存在任一 voxel 為 3 → label=1，否則 label=0

為避免類別不平衡，本研究採用：

正樣本：全部 10 位  

負樣本：由 40 位中抽樣 20 位（總計 30 位）

##### (2) 輸入影像縮放避免 OOM

由於整張 CT 解析度高（如 512×512×數百 slices），若直接輸入 3D CNN 會造成 GPU OOM。

因此程式在 transform 中加downsample : ResizeD(spatial_size=(128,128,128))

確保分類訓練能在 16GB GPU 上穩定執行。

##### (3) 模型與 Loss

模型：MONAI resnet.resnet10  

Loss：Cross Entropy  

Optimizer：AdamW  

---

### 四、infer_and_dice.py

##### (1) 權重檔名與載入

class12：best_swinunetr_class12.pth  

class3：best_swinunetr_class3.pth  

classifier：best_class3_classifier.pth  

推論端以 argparse 參數統一管理並載入。

##### (2) 兩分支前處理差異

class12：含 Spacing(1,1,1)（與其訓練一致）  

class3：保留原 spacing（與其訓練一致）

##### (3) 融合規則

final 以 class12 預測為底，若 class3 分支預測為 1 的 voxel，覆蓋成 label=3（class3 優先覆寫）

github連結：https://github.com/f114152115-creator/NKUST-/tree/main

---

### 二、模型訓練流程

#### (一) 訓練階段:

1.掃描標註資料，辨識含有 class3 的病人  

2.訓練 class1 + class2 分割模型（主結構）  

3.訓練 class3 專用 tiny lesion segmentation 模型  

4.訓練病人層級 class3 classifier（輔助模組）

#### (二) 推論階段：

1.推論class1 + class2 分割模型  

2.推論class3 專用 tiny lesion segmentation 模型  

3.推論病人層級 class3 classifier  

4.用class3 classifier是否要將class3 專用 tiny lesion segmentation 模型融合至class1 + class2 分割模型結果中

---

### 三、參數設定

模型任務 | 項目 | 設定內容
--- | --- | ---
Class1 + Class2 Segmentation | Patch size | 96³ / 128³
 | Batch size | 1
 | Optimizer | AdamW
 | Learning rate | 1 × 10⁻⁴
 | Loss function | Dice Loss + Cross Entropy Loss
 | Early stopping | Validation Dice
 | 評估重點 | 大型解剖結構之穩定分割
Class3 Segmentation（Tiny Lesion） | Patch size | 64³ ~ 96³
 | Batch size | 1
 | Loss function | Dice-based Loss（偏向 Recall）
 | 評估指標 | Recall
 |  | Precision
 |  | Lesion-level F1
 | Early stopping | (Recall + Lesion-F1) 組合分數
 | 設計重點 | 提升極小病灶之偵測能力
Class3 Classifier（病人層級） | Input size | 128 × 128 × 128（downsample）
 | Batch size | 1
 | Optimizer | AdamW
 | Loss function | Cross Entropy Loss
 | 評估指標 | Patient-level Accuracy
 | 使用定位 | Segmentation pipeline 之輔助分析模組

---

## 參、分析與結論

針對心臟電腦斷層（Cardiac CT）多類別分割任務中，不同解剖結構在空間尺度與出現頻率上高度不平衡的問題，本研究提出一套多模型分流式（multi-model decoupled）segmentation pipeline，藉由將結構性質明顯不同的目標分別建模，以提升整體訓練穩定性與分割效能。傳統單一模型同時處理大型解剖結構與極小病灶時，模型容易受到資料分佈不均與損失函數偏向大體積結構的影響，導致微小病灶在訓練過程中被忽略，甚至無法有效收斂。本研究的分流策略正是針對此一核心問題所設計。

實驗結果顯示，將主要解剖結構（心肌與主動脈瓣）與極小病灶（鈣化，class3）分開訓練，能夠顯著提升模型的訓練穩定性。對於體積較大且形態穩定的結構，模型可在較一致的資料分佈下學習空間特徵，避免受到 tiny lesion 所造成的梯度不穩定影響；而對於 class3 鈣化病灶，獨立建模後得以採用更適合極小目標的訓練策略與資料抽樣方式，使模型能有效聚焦於病灶特徵本身。

在評估指標方面，本研究進一步驗證了 Dice coefficient 並不適合單獨作為 tiny lesion 分割的主要評估依據。由於鈣化病灶體積極小，即使模型成功偵測到病灶，其 Dice 分數仍可能偏低，難以反映實際臨床意義。因此，本研究改以 Recall、Precision 及 lesion-level 指標作為 class3 分割效能的主要評估方式，其中 Recall 與 lesion-level 指標更能反映模型是否成功偵測病灶，對臨床應用而言具有更高參考價值。

此外，本研究亦嘗試引入病人層級的 class3 classifier，用以判斷整體影像中是否存在鈣化病灶，作為 segmentation pipeline 的輔助模組。實驗結果顯示，在病人數量有限的情況下，該分類模型容易出現過擬合，於驗證資料上的表現亦較不穩定，顯示僅依賴 classifier 作為最終決策依據仍具風險。然而，其結果仍可作為 segmentation 輸出的輔助參考，提供額外的分析視角。

綜合上述結果，本研究最終採用 以 segmentation 輸出為基礎的 presence 判斷策略 作為系統核心，透過 class3 segmentation 是否產生有效病灶預測來判定病人是否存在鈣化，兼顧保守性與實務可行性。整體而言，本研究所提出之多模型分流式方法，在有限資料條件下有效改善了多尺度結構分割的困難，並為心臟 CT 中結構尺度高度不平衡的分割任務提供一具可行且具臨床意義的解決方案。

---

## 肆、使用的外部資源與參考文獻

[1] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS).

[2] Cardoso, M. J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., … Ourselin, S. (2022). MONAI: An open-source framework for deep learning in healthcare. Medical Image Analysis, 73, 102154. https://doi.org/10.1016/j.media.2021.102154

[3] OpenAI. (2023). ChatGPT (GPT-4) [Large language model]. https://www.openai.com/
