# 🦠 HW1-Regression COVID Prediction

## 📖 Introduction

[HW1](https://www.kaggle.com/competitions/ml2023spring-hw1/overview) 的任务是根据美国某个州**过去三天**的调查结果，预测第三天的
**新增检测阳性病例**的百分比。

![task](misc/hw1-task.png)

训练数据有 3009 条，测试数据为 997 条。使用均方误差 (MSE) 作为评估指标。其中每一条包含 88 个 features，
每个 feature 详情如下：

- id (1)
- States (34，使用 one-hot 编码)
- **Survey (18 * 3 days)**
    - COVID-like illness (5)
        - cli, ili …
    - Behavior indicators (5)
        - wearing_mask, shop_indoors, restaurant_indoors, public_transit …
    - Belief indicators (2)
        - belief_mask_effective, belief_distancing_effective.
    - Mental indicator (2)
        - worried_catch_covid, worried_finance.
    - Environmental indicators (3)
        - other_masked_public, other_distanced_public …
- **Tested Positive Cases (1 * 3 days)**
    - tested_positive (第三天的 tested_positive 是需要我们预测的)

## 🎯 Baseline

|        | Score   | Hint                                                                                      | Public | Private |
|--------|---------|-------------------------------------------------------------------------------------------|--------|---------|
| simple | 1.96993 | Just run [sample code](https://www.kaggle.com/code/b08502105/ml2023spring-hw1-samplecode) | ✅      | ✅       |
| medium | 1.15678 | Feature selection                                                                         | ✅      | ✅       |                                                                                                                                               
| strong | 0.92619 | Different optimizers and L2 regularization                                                | ✅      | ✅       |
| boss   | 0.81456 | Better feature selection, different model architectures and try more hyperparameters      | ✅      | ❌       |

## 🕹️ Quick Start

下载数据集：

```bash
wget https://github.com/hsushuai/ml2023spring-hw/releases/download/dataset/ml2023spring-hw1.zip

unzip ml2023spring-hw1.zip
```

运行 hw1：

```bash
python main.py hw1 --data_dir YOUR_DATA_DIRECTORY --output YOUR_OUTPUT_DIRECTORY
```

你需要将 `YOUR_DATA_DIRECTORY` 和 `YOUR_OUTPUT_DIRECTORY` 替换成实际的数据目录和输出目录，默认为 'data/ml2023spring-hw1'
和 'output'。

❗ 注意，请确保数据目录结构如下：

```text
data_dir/
│
├── covid_train.csv
└── covid_test.csv
```

## 📕 Docs

### 💯 Leaderboard Score

在完成 HW1 时，我们做了大量尝试，最终只有 Public Score 过了 Boss Baseline，Private 只有过 Strong。

![score](misc/hw1-score.png)

### 🗳️ Feature Select

特征选择参考了 [a86gj387](https://zhuanlan.zhihu.com/p/483652591) 的方法，
计算 Feature 和 tested_positive 的相关度，只选择相关度 $>0.5$ 部分的 17 个 特征。

### 🧬 Network Architecture

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 304.5616771674445 312.7027373988958" width="304.5616771674445" height="312.7027373988958">
  <!-- svg-source:excalidraw -->

  <defs>
    <style class="style-fonts">
      @font-face {
        font-family: "Virgil";
        src: url("https://excalidraw.com/Virgil.woff2");
      }
      @font-face {
        font-family: "Cascadia";
        src: url("https://excalidraw.com/Cascadia.woff2");
      }
      @font-face {
        font-family: "Assistant";
        src: url("https://excalidraw.com/Assistant-Regular.woff2");
      }
    </style>

  </defs>
  <rect x="0" y="0" width="304.5616771674445" height="312.7027373988958" fill="#ffffff"></rect><g stroke-linecap="round" transform="translate(19.584486905654103 198.6183937213823) rotate(0 118.48408812679247 21.647720424839463)"><path d="M10.82 0 C61.6 0.87, 115.11 0.37, 226.14 0 M10.82 0 C86.34 0.64, 161.71 -0.27, 226.14 0 M226.14 0 C233.08 1.52, 236.75 3.2, 236.97 10.82 M226.14 0 C231.43 -0.42, 238.84 2.95, 236.97 10.82 M236.97 10.82 C236.81 19.13, 237.77 28.09, 236.97 32.47 M236.97 10.82 C236.13 19.1, 237.24 25.73, 236.97 32.47 M236.97 32.47 C236.1 39.11, 233.76 45.16, 226.14 43.3 M236.97 32.47 C237.98 38.04, 233.58 44.71, 226.14 43.3 M226.14 43.3 C144.52 41.65, 67.15 41.58, 10.82 43.3 M226.14 43.3 C177.26 44.05, 128.47 43.88, 10.82 43.3 M10.82 43.3 C2.21 42.46, 1.75 38.99, 0 32.47 M10.82 43.3 C5.87 44.01, 0.79 39.98, 0 32.47 M0 32.47 C-1.7 25.66, 0.81 17.11, 0 10.82 M0 32.47 C0.72 24.8, -0.17 17.82, 0 10.82 M0 10.82 C1.16 1.64, 2.19 -1.26, 10.82 0 M0 10.82 C1.76 4.41, 5.65 2.24, 10.82 0" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(92.31028188059122 212.3940562560515) rotate(0 45.75829315185547 7.872057890170254)"><text x="45.75829315185547" y="0" font-family="Virgil, Segoe UI Emoji" font-size="12.59529262427247px" fill="#1e1e1e" text-anchor="middle" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">LazyLinear(64)</text></g><g stroke-linecap="round" transform="translate(19.584486905654103 137.0086750682599) rotate(0 118.48408812679247 21.647720424839463)"><path d="M10.82 0 C83.59 -1.08, 154.85 0.63, 226.14 0 M10.82 0 C96.53 0.76, 182.38 -0.09, 226.14 0 M226.14 0 C231.68 -0.37, 238.59 3.04, 236.97 10.82 M226.14 0 C235.61 0.62, 234.74 3.36, 236.97 10.82 M236.97 10.82 C238.33 17.32, 238.05 21.39, 236.97 32.47 M236.97 10.82 C236.59 17.88, 237.22 26.45, 236.97 32.47 M236.97 32.47 C237.85 38.25, 233.55 44.52, 226.14 43.3 M236.97 32.47 C238.61 39.49, 232.98 41.13, 226.14 43.3 M226.14 43.3 C176.7 43.05, 130.24 41.5, 10.82 43.3 M226.14 43.3 C153.29 43.24, 82.41 43.3, 10.82 43.3 M10.82 43.3 C5.57 43.92, 0.69 39.94, 0 32.47 M10.82 43.3 C4.01 43.4, 0.96 37.74, 0 32.47 M0 32.47 C0.74 25.08, -1.79 20.62, 0 10.82 M0 32.47 C0.64 25.59, -0.65 20.05, 0 10.82 M0 10.82 C1.53 4.3, 5.38 1.95, 10.82 0 M0 10.82 C1.8 3.4, 1.46 -1.85, 10.82 0" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(104.66105092355997 150.7843376029291) rotate(0 33.40752410888672 7.872057890170254)"><text x="33.40752410888672" y="0" font-family="Virgil, Segoe UI Emoji" font-size="12.59529262427247px" fill="#1e1e1e" text-anchor="middle" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">LeakyReLU</text></g><g stroke-linecap="round" transform="translate(19.584486905654103 62.41262556931406) rotate(0 118.48408812679247 21.647720424839463)"><path d="M10.82 0 C65.06 0.39, 115.75 -1.03, 226.14 0 M10.82 0 C91.09 0.22, 173.53 0.13, 226.14 0 M226.14 0 C235.32 0.54, 235.03 3.39, 236.97 10.82 M226.14 0 C233 0.86, 237.02 5.08, 236.97 10.82 M236.97 10.82 C237.86 18.15, 235.37 23.68, 236.97 32.47 M236.97 10.82 C237.45 15.98, 237.11 23.18, 236.97 32.47 M236.97 32.47 C238.4 39.52, 233.03 41.41, 226.14 43.3 M236.97 32.47 C235.21 40.94, 231.51 41.6, 226.14 43.3 M226.14 43.3 C146.5 42.62, 65.9 41.77, 10.82 43.3 M226.14 43.3 C171.45 42.91, 115.33 42.75, 10.82 43.3 M10.82 43.3 C3.95 43.39, 0.84 37.99, 0 32.47 M10.82 43.3 C3.67 44.24, -1.87 40.64, 0 32.47 M0 32.47 C-0.37 26.37, 0.46 22.45, 0 10.82 M0 32.47 C0.92 28.46, 1.04 24.72, 0 10.82 M0 10.82 C1.56 3.43, 1.74 -1.61, 10.82 0 M0 10.82 C0.23 3.63, 4.14 0.94, 10.82 0" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(98.66192830393106 76.18828810398327) rotate(0 39.406646728515625 7.872057890170254)"><text x="39.406646728515625" y="0" font-family="Virgil, Segoe UI Emoji" font-size="12.59529262427247px" fill="#1e1e1e" text-anchor="middle" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">LazyLinear(1)</text></g><g stroke-linecap="round" transform="translate(10 121.37834783466951) rotate(0 128.06857503244657 68.08290698499081)"><path d="M32 0 C104.12 0.05, 172.78 -1.41, 224.14 0 M224.14 0 C245.1 1.63, 255.57 12.62, 256.14 32 M256.14 32 C255.13 56.35, 255.48 80.48, 256.14 104.17 M256.14 104.17 C256.47 126.5, 243.72 136.95, 224.14 136.17 M224.14 136.17 C161.42 135.73, 101.73 136.21, 32 136.17 M32 136.17 C9.23 136.36, 1.23 126.93, 0 104.17 M0 104.17 C-1.41 84.22, -1.49 60.62, 0 32 M0 32 C1.46 11.13, 10.35 0.71, 32 0" stroke="#2f9e44" stroke-width="2.5" fill="none" stroke-dasharray="8 10"></path></g><g stroke-linecap="round"><g transform="translate(130.96283474032384 198.05424307990188) rotate(0 0 -9.088737582947715)"><path d="M-0.06 -0.11 C-0.12 -3.23, -0.55 -15.24, -0.46 -18.28 M-0.75 -0.65 C-0.65 -3.71, 0.36 -14.77, 0.54 -17.63" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(130.96283474032384 198.05424307990188) rotate(0 0 -9.088737582947715)"><path d="M2.97 -8.88 C2.15 -11.76, 1.46 -13.52, 0.54 -17.63 M2.97 -8.88 C2.05 -11.27, 1.57 -13.54, 0.54 -17.63" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(130.96283474032384 198.05424307990188) rotate(0 0 -9.088737582947715)"><path d="M-3.22 -9.36 C-2.27 -12.1, -1.19 -13.72, 0.54 -17.63 M-3.22 -9.36 C-2.46 -11.63, -1.26 -13.77, 0.54 -17.63" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(128.97983744949875 135.92032796738613) rotate(0 -0.49574932270627414 -15.368229003893475)"><path d="M-0.46 -0.1 C-0.54 -5.23, -0.55 -25.81, -0.54 -30.89 M0.3 -0.63 C0.19 -5.67, -0.61 -25.39, -0.75 -30.27" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(128.97983744949875 135.92032796738613) rotate(0 -0.49574932270627414 -15.368229003893475)"><path d="M5.02 -16.02 C3.22 -19.54, 1.48 -23.54, -0.75 -30.27 M5.02 -16.02 C3.14 -20.54, 1.33 -25.1, -0.75 -30.27" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(128.97983744949875 135.92032796738613) rotate(0 -0.49574932270627414 -15.368229003893475)"><path d="M-5.49 -15.65 C-4.25 -19.23, -2.95 -23.34, -0.75 -30.27 M-5.49 -15.65 C-3.93 -20.29, -2.31 -24.97, -0.75 -30.27" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(133.27633157961964 279.3571320037263) rotate(0 -0.49574932270627414 -18.012225391660195)"><path d="M0.45 -0.16 C0.45 -6.12, -0.12 -29.89, -0.45 -35.88 M0.01 -0.71 C0.03 -6.84, -0.53 -30.96, -0.61 -36.77" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(133.27633157961964 279.3571320037263) rotate(0 -0.49574932270627414 -18.012225391660195)"><path d="M5.87 -19.95 C4.23 -22.68, 3.33 -28.07, -0.61 -36.77 M5.87 -19.95 C4.02 -24.51, 2.21 -29.44, -0.61 -36.77" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(133.27633157961964 279.3571320037263) rotate(0 -0.49574932270627414 -18.012225391660195)"><path d="M-6.45 -19.72 C-5.43 -22.55, -3.67 -27.99, -0.61 -36.77 M-6.45 -19.72 C-4.74 -24.37, -2.98 -29.37, -0.61 -36.77" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g></g><mask></mask><g transform="translate(99.89587718406574 286.9586216185553) rotate(0 33.25156217123231 7.872057890170254)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="12.59529262427247px" fill="#1e1e1e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">input data</text></g><g stroke-linecap="round"><g transform="translate(126.99684015867388 62.21892865839118) rotate(0 -0.1652497742354626 -17.020726746247647)"><path d="M-0.53 -0.06 C-0.69 -5.72, -0.45 -28.23, -0.42 -33.84 M0.2 -0.57 C-0.09 -6.41, -0.75 -29.03, -0.89 -34.7" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(126.99684015867388 62.21892865839118) rotate(0 -0.1652497742354626 -17.020726746247647)"><path d="M5.37 -18.87 C2.77 -25.33, 0.69 -31.47, -0.89 -34.7 M5.37 -18.87 C2.71 -24.45, 0.9 -30.74, -0.89 -34.7" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g><g transform="translate(126.99684015867388 62.21892865839118) rotate(0 -0.1652497742354626 -17.020726746247647)"><path d="M-6.27 -18.55 C-4.24 -25.08, -1.69 -31.35, -0.89 -34.7 M-6.27 -18.55 C-4.69 -24.27, -2.27 -30.68, -0.89 -34.7" stroke="#1e1e1e" stroke-width="2" fill="none"></path></g></g><mask></mask><g transform="translate(101.87887447489061 10) rotate(0 25.253553810566586 7.872057890170254)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="12.59529262427247px" fill="#1e1e1e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">softmax</text></g><g transform="translate(274.0691392281931 182.85126385024387) rotate(0 10.246268969625703 7.872057890170254)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="12.59529262427247px" fill="#2f9e44" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">× 3</text></g></svg>

### ⚙️ Configs

|          | Parameter     | Value  |
|----------|---------------|--------|
| Model    | num_layers    | 5      |
|          | hidden_size   | 64     |
| Training | batch_size    | 256    |
|          | max_epochs    | 1000   |
|          | learning_rate | 0.001  |
|          | weight_decay  | 0.0001 |
|          | early_stop    | 600    |

更多详细的设置请参考源代码 [hw1-configs](../configs/hw1-config.yaml)。

### 🎭 Tricks

这里的优化器使用了 Adam。

## 🙌 Contribute

虽然最终的 Public Score 达到了 Strong，但是 Private Score 还是差了不少。如果你有更好的 Solution 欢迎分享。
或者如果你遇到了什么问题，欢迎提交 issue。
