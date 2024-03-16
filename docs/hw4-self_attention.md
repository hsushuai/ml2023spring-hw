# 🗣️ HW4-Self-Attention: Speakers Classification

## 📖 Introduction

[HW4](https://www.kaggle.com/competitions/ml2023springhw4) 的任务是识别说话的人分类，共 **600 个类别**。

和作业二一样，TA 已经将语音做了预处理 (waveforms -> mel-spectrogram)，我们只需加载处理好的 tensor 即可。数据文件目录详情如下：

- `metadata.json`: 记录关于特征的信息。在 `speakers` 关键字下保存了 600 个 speaker 的 id 以及其对应的多段语音预处理后的特征 tensor 文件路径
- `testdata.json`: 记录了 8000 条语音特征文件路径
- `mapping.json`: 记录了两个字典，`speakers2id` 和 `id2speakers` 分别记录 speakers（作业提交文件中的 id 项） 和其 0~599 (模型输出的预测) 的映射
- `uttr-{audioID}.pt`: 56666 (train & valid) + 8000 (test) 条特征

> 在加载训练数据时，我们需要先读取 `metadata.json` 保存所有的 `feature_path` 及其对应地标签。需要注意的是，每条特征路径对应地标签不是 `metadata.json` 文件中的 `id*****` 而是在 `mapping.json` 文件中 `id*****` 对应的 `0~599` 标签。

## 🎯 Baseline

|        | Public Baseline | Hints                                                                                        | Estimate Training Time | Public | Private |
| ------ | --------------- | -------------------------------------------------------------------------------------------- | ---------------------- | ------ | ------- |
| Simple | 0.66025         | Run Sample Code.                                                                             | 30~40 mins on Colab    | ✅      | ✅       |
| Medium | 0.81750         | Modify the parameters of the transformer modules in the sample code.                         | 1~1.5 hour on Colab    | ✅      | ✅       |
| Strong | 0.88500         | Construct Conformer, which is a variety of Transformer.                                      | 1~1.5 hour on Colab    | ✅      | ✅       |
| Boss   | 0.93000         | Implement Self-Attention Pooling & Additive Margin Softmax to further boost the performance. | 40+hr on Kaggle        | ✅      | ✅       |

## ⚡ Quick Start

下载数据集：

```bash
wget https://github.com/hsushuai/ml2023spring-hw/releases/download/dataset/ml2023spring-hw4.tar.gz.part-a*

cat ml2023spring-hw4.tar.gz.part-a* > ml2023spring-hw4.tar.gz
tar -zxvf ml2023spring-hw4.tar.gz
```

运行 hw4：

```bash
python main.py hw4 --data_dir YOUR_DATA_DIRECTORY --output YOUR_OUTPUT_DIRECTORY
```

你需要将 `YOUR_DATA_DIRECTORY` 和 `YOUR_OUTPUT_DIRECTORY` 替换成实际的数据目录和输出目录，默认为 'data/ml2023spring-hw4'
和 'output'。

❗ 注意，请确保数据目录结构如下：

```text
data_dir/
│
├── mapping.json
├── metadata.json
├── testdata.json
└── uttr-{audioID}.pt
```

## 📕 Docs

### Leader board Score

![score](misc/hw4-score.png)

### Network Architecture

- **Conformer** + Self-Attention Pooling + Additive Margin Softmax

### Configs

| Section  | Parameter                | Value  |
| -------- | ------------------------ | ------ |
| data     | valid_ratio              | 0.1    |
|          | segment_len              | 128    |
| model    | d_model                  | 80     |
| training | batch_size               | 64     |
|          | learning_rate            | 0.001  |
|          | dropout                  | 0.1    |
|          | weight_decay             | 0.0001 |
|          | max_steps                | 70000  |
|          | validation_after_n_steps | 2000   |
|          | save_best_freq           | 10000  |
|          | warmup_steps             | 1000   |

更多详细配置请参考源代码 [hw4-configs](../configs/hw4-config.yaml)。

## 🎭 Tricks

- 使用 AdamW 作优化器
- hw4 的训练并没有按照之前以 epoch 为单位训练，而是直接以 step 为单位，每个 step 反向更新后使用 warmup 和 cosine lr schedule

## 🙌 Contribute

如果你有更好的 Solution 欢迎分享。或者如果你遇到了什么问题，欢迎提交 issue。
