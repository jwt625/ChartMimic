<div align="center">
<img src="./assets/chartmimic.png" style="width: 20%;height: 10%">
<h1> ChartMimic: Evaluating LMM’s Cross-Modal Reasoning Capability via Chart-to-Code Generation
 </h1>
</div>

<div align="center">

![Data License](https://img.shields.io/badge/Data%20License-Apache--2.0-blue.svg)
![Code License](https://img.shields.io/badge/Code%20License-Apache--2.0-blue.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9.0-blue.svg)
</div>

<div align="center">
  <!-- <a href="#model">Model</a> • -->
  🌐 <a href="https://chartmimic.github.io/">Website</a> |
  🏆 <a href="https://chartmimic.github.io/">Leaderboard</a> |
  📚 <a href="https://huggingface.co/datasets/ChartMimic/ChartMimic">Data</a> |
  📃 <a href="">Paper</a>
</div>


## What's New
- **[2024.06.13]** 📣 ChartMimic is released.

## Introduction

ChartMimic aims at assessing **the visually-grounded code generation capabilities** of large multimodal models (LMMs). ChartMimic utilizes information-intensive visual charts and textual instructions as inputs, requiring LMMs to generate the corresponding code for chart rendering.

ChartMimic includes **1,000 human-curated (figure, instruction, code) triplets**, which represent the authentic chart use cases found in scientific papers across various domains(e.g., Physics, Computer Science, Economics, etc). These charts span 18 regular types and 4 advanced types, diversifying into 191 subcategories. Furthermore, we propose **multi-level evaluation metrics** to provide an automatic and thorough assessment of the output code and the rendered charts. Unlike existing code generation benchmarks, ChartMimic places emphasis on evaluating LMMs' capacity to harmonize a blend of cognitive capabilities, encompassing **visual understanding, code generation, and cross-modal reasoning**.

<div align="center">
<img src="./assets/framework.png" style="width: 100%;height: 100%">
</div>


## Table of Contents
<details>
<summary>
Click to expand the table of contents
</summary>

- [What's New](#whats-new)
- [Introduction](#introduction)
- [🚀 Quick Start](#-quick-start)
  - [Setup Environment](#setup-environment)
  - [Evaluate Models](#evaluate-models)
- [📚 Data](#data)
  - [Data Overview](#data-overview)
  - [Download Link](#download-link)
- [️Citation](#️citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
</details>


## 🚀 Quick Start 

Here we provide a quick start guide to evaluate LMMs on ChartMimic.

### Setup Environment
```shell
conda create -n chartmimic python==3.9.0
conda activate chartmimic
pip install -r requirements.txt
```

### Download Data
You can download the whole evaluation data by running the following command:
```shell
cd ChartMimic # cd to the root directory of this repository
mkdir dataset
wget https://huggingface.co/datasets/ChartMimic/ChartMimic/blob/main/test.tar.gz
tar -xzvf filename.tar.gz -C dataset
```

### Evaluate Models
#### Direct Mimic
Example script for `gpt-4-vision-preview` on the `Direct Mimic` task:
```
python chart2code/main.py \
--cfg_path eval_configs/direct_generation.yaml \
--tasks chart2code \
--model gpt-4-vision-preview
```

#### Customized Mimic
Example script for `gpt-4-vision-preview` on the `Customized Mimic` task:
```
python chart2code/main.py \
--cfg_path eval_configs/edit_direct.yaml \
--tasks chartedit \
--model gpt-4-vision-preview
```

#### Different LMMs
We now offer configuration for 14 SOTA LMM models (`gpt-4-vision-preview`, `claude-3-opus-20240229`, `gemini-pro-vision`, `Phi-3-vision-128k-instruct`,`MiniCPM-Llama3-V-2_5`,`InternVL-Chat-V1-5`, `cogvlm2-llama3-chat-19B`,`deepseekvl`,`llava-v1.6-mistral-7b-hf`,`llava-v1.6-34b-hf`, `idefics2-8b`, `llava-v1.6-vicuna-13b-hf`,`llava-v1.6-vicuna-7b-hf` and `qwenvl`).
 <!-- and a simple agent based on direct prompting. You could also customize your own [agents](https://github.com/hkust-nlp/AgentBoard/blob/main/assets/agent_customization.md) and [LMMs](https://github.com/hkust-nlp/AgentBoard/blob/main/assets/llm_customization.md). Models supported by [vLLM](https://github.com/vllm-project/vllm) should be generally supported in AgentBoard, while different models may require specific prompt templates. -->

<!-- #### Different Prompting Methods -->

## 📚 Data
### Download Link
You can download the whole evaluation data by running the following command:
```shell
wget https://huggingface.co/datasets/ChartMimic/ChartMimic/blob/main/dataset.zip
```
Please uncommpress the file and move the data to `ChartMimic/dataset`.
```shell
cd ChartMimic
mkdir dataset
upzip dataset.zip
```


## Citation
If you find this repository useful, please consider giving star and citing our paper:
```
@article{shi2024chartmimic,
  title={ChartMimic: Evaluating LMM’s Cross-Modal Reasoning Capability via Chart-to-Code Generation},
  author={Chufan Shi and Cheng Yang and Yaxin Liu and Bo Shui and Junjie Wang and Mohan Jing and Linran
Xu and Xinyu Zhu and Siheng Li and Yuxiang Zhang and Gongye Liu and Xiaomei Nie and Deng Cai and Yujiu
Yang},
  year={2024},
}
```


## License
[![Apache-2.0 license](https://img.shields.io/badge/Code%20License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

The ChartMimic data and codebase is licensed under a [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).


## Acknowledgements
We would like to express our gratitude to [agentboard](https://github.com/hkust-nlp/AgentBoard) for their project codebase.

<!-- ## Scaffold Agent
1. Generate dot picture
```shell
python chart2code/utils/data_process/dot_processor.py
```

2. Run chart2code task
```shell
bash run.sh
```# ChartMimic -->
