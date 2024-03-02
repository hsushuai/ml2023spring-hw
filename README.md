# Solutions for Spring 2023 Machine Learning (taught by Prof. Hung-yi Lee) course assignments.

## 📖 Homework List

| Completed | #                                  | Topic                  | Task             | Public Baseline | Private Baseline |
|-----------|------------------------------------|------------------------|------------------|-----------------|------------------|
| ✅         | [HW 1](docs/hw1-regression.md)     | Regression             | 预测 COVID-19 新增病例 | Boss            | Strong           |
| ✅         | [HW 2](docs/hw2-classification.md) | Classification         | Phoneme 分类       | Strong          |                  |
|           | HW 3                               | CNN                    |                  |                 |                  |
|           | HW 4                               | Self-attention         |                  |                 |                  |
|           | HW 5                               | Transformer            |                  |                 |                  |
|           | HW 6                               | Generative Model       |                  |                 |                  |
|           | HW 7                               | BERT                   |                  |                 |                  |
|           | HW 8                               | Auto-encoder           |                  |                 |                  |
|           | HW 9                               | Explainable AI         |                  |                 |                  |
|           | HW 10                              | Attack                 |                  |                 |                  |
|           | HW 11                              | Adaptation             |                  |                 |                  |
|           | HW 12                              | Reinforcement Learning |                  |                 |                  |
|           | HW 13                              | Network Compression    |                  |                 |                  |
|           | HW 14                              | Life-long Learning     |                  |                 |                  |
|           | HW 15                              | Meta Learning          |                  |                 |                  |

## ⚡ Quick Start

```bash
https://github.com/hsushuai/ml2023spring-hw.git

cd ml2023spring

pip install -r requirements.txt
```

### Download Data

在 [releases](https://github.com/hsushuai/ml2023spring-hw/releases) 中下载对应作业的数据文件，并解压。
或者复制对应的下载连接，使用 `wget` 下载，以 HW 1 为例：

```bash
wget https://github.com/hsushuai/ml2023spring-hw/releases/download/dataset/ml2023spring-hw1.zip

unzip ml2023spring-hw1.zip
```

### Running

以 HW 1 为例：

```bash
python main.py hw1
```

运行命令**必须指定需要运行的作业**，可以是从 `hw1` 一直到 `hw15`。

此外，你还可以在命令中添加可选参数，参数名称为对应作业中对应的 [configs](configs)。以 HW 1 为例，可以在命令中任意设置 [hw1-config](configs/hw1-config.yaml)
中的参数，比如：

```bash
python main.py hw1 --max_epochs 3000 --data_dir /mnt/data --output_dir /mnt/output
```



