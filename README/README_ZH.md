<div align="center">
 <picture>
    <source media="(prefers-color-scheme: light)" srcset="asserts/ETrajEval_logo_white.png">
    <source media="(prefers-color-scheme: dark)" srcset="asserts/ETrajEval_logo_black.png">
    <img alt="ETrajEval logo" width="700" src="asserts/ETrajEval_logo.png">
    </picture>
  <br/>
  <p>
    <!-- <a href="?"><img alt="Website" src="https://img.shields.io/badge/🌐_Project-Website-A593C2?style=flat-square&labelColor=8A7AA8"></a> -->
    <!-- <a href="?"><img alt="Paper" src="https://img.shields.io/badge/📄_arXiv-Paper-C7969C?style=flat-square&labelColor=A8798A"></a> -->
    <a href="https://pypi.org/project/rewardanything/"><img alt="Apache" src="https://img.shields.io/badge/⚖️License-Apache_2.0-7B9BB3?style=flat-square"></a>
    </p>

  # Detecting Emotional Dynamic Trajectories: An Evaluation Framework for Emotional Support in Language Models
    
  <a>Zhouxing Tan<sup>1</sup></a>&emsp;
  <a>Ruochong Xiong<sup>1</sup></a>&emsp;
  <a>Yulong Wan<sup>1</sup></a>&emsp;
  <a>JinLong Ma<sup>2</sup></a>&emsp;
  <a>Hanlin Xue<sup>1</sup></a>&emsp;
  <a>Qichun Deng<sup>2</sup></a>&emsp;
  <a>Haifeng Jing<sup>1</sup></a>&emsp;
  <a>Zhengtong Zhang<sup>2</sup></a>&emsp;
  <a>Depei Liu<sup>1</sup></a>&emsp;
  <a>Shiyuan Luo<sup>2</sup></a>&emsp;
  <a> Junfei Liu<sup>1,†</sup></a>
  <div>
    <br/>
    <p>
      <sup>1</sup>National Engineering Research Center For Software Engineering, Peking University&emsp;
      <sup>2</sup>Guangzhou Quwan Network Technology
    </p>
    <p><sup>†</sup>Corresponding author.</p>
  </div>
</div>


本项目是 **ETrajEval** 的官方代码仓库，该框架源自我们的论文 **《Detecting Emotional Dynamic Trajectories: An Evaluation Framework for Emotional Support in Language Models》**。

传统的情感支持评估方法通常依赖于简短、静态的对话，在孤立的片段中评估模型的回应。这种方法未能捕捉到有效支持的本质，即支持能力是在时间中展开的，并且需要适应用户不断变化的情感状态。

为了克服这一局限，**ETrajEval** 从静态的、以模型为中心的评估转向**动态的、以用户为中心的轨迹分析**。我们模拟了真实的多轮角色扮演场景，在这些场景中，用户的情感状态会受到对话和外部“干扰事件”的影响。我们的框架评估的不仅仅是模型生成单个共情回应的能力，更是在整个互动过程中持续**改善和稳定用户情绪健康**的能力。


<p align="center">
  <img src="asserts/ETrajEval_main.png" width="800" alt="框架概览">
  <br>
  <em>图1: ETrajEval 框架概览，包括动态交互、因果情感估计和基于轨迹的度量指标。</em>
</p>

## 🙂 语言

<center> 

[英文](../README.md)   | 中文

</center>

## 🌟 核心特性

-   🧠 **动态长期评估**: 模拟多轮对话，以评估持续的情感支持能力，超越了单轮问答的限制。
-   📈 **以用户为中心的轨迹度量指标**: 引入了三个创新的度量指标来量化情感动态：
    -   **BEL (Baseline Emotional Level)**: 互动期间用户的平均情感水平。
    -   **ETV (Emotional Trajectory Volatility)**: 模型提升负面情绪和抵抗情绪恶化的效率。
    -   **ECP (Emotional Centroid Position)**: 情感转变的总体方向和稳定性。
-   🌍 **真实的模拟环境**: 基于一个大规模基准构建，包含 **328个情感背景** 和 **1152个干扰事件**，并以成熟的心理学理论为基础。
-   🔬 **因果校准的情感评分**: 采用因果推断技术，获得对用户情感状态的无偏、鲁棒的估计，减轻混淆因素的影响。
-   🔧 **可配置与可扩展**: 使用单个YAML文件即可轻松配置和比较不同的模型（本地或基于API）。
-   📊 **全面的分析报告**: 生成详细的逐轮评分数据和聚合结果，为模型性能提供深刻的洞见。

## 🚀 快速上手

### 1. 环境设置

首先，克隆本仓库并设置环境。

```bash
# 克隆本仓库
git clone https://github.com/your-username/ETrajEval.git
cd ETrajEval

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**关于PyTorch的说明**: 为获得GPU支持，请安装与您系统CUDA工具包相匹配的PyTorch版本。请访问 [PyTorch官方网站](https://pytorch.org/get-started/locally/) 以获取正确的安装命令。

### 2. 下载模型

该框架在评分阶段需要一个奖励模型。请下载您在配置文件中指定的模型，并更新路径。例如：

```bash
# 示例：从Hugging Face下载奖励模型
# 您需要先安装git-lfs: https://git-lfs.com
git clone https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B /path/to/your/models/Skywork-Reward-V2-Llama-3.1-8B
```

### 3. 配置评估任务

修改 `ETrajEval/config.yaml` 文件来设置您的实验。这是整个框架的中央控制文件。

```yaml
# config.yaml

# 控制运行哪个阶段
execution_phases:
  run_generation: true  # 阶段一：模拟对话
  run_scoring: true     # 阶段二：为生成的对话评分

# --- 配置模型1 (例如，您要测试的模型) ---
model1:
  name: "My-Custom-Model-7B"
  type: "local" # 'local' 代表本地HuggingFace模型, 'api' 代表兼容OpenAI的API模型
  path_or_name: "/path/to/your/models/My-Custom-Model-7B"
  gpu_indices: [0] # 使用的GPU索引列表
  generation_params:
    temperature: 1.0
    max_tokens: 512

# --- 配置模型2 (常用通用聊天模型) ---
model2:
  name: "gpt-4-turbo"
  type: "api"
  model_name: "gpt-4-turbo-2024-04-09"
  api_config:
    api_key: "YOUR_API_KEY_HERE" # 或者设置为 OPENAI_API_KEY 环境变量
    base_url: "https://api.openai.com/v1"
  generation_params:
    temperature: 1.0
    max_tokens: 512

# --- 数据与评分配置 ---
data:
  prompt_file: "dataset/prompts.json" # 对话场景文件路径

scoring:
  reward_model_device: "cuda:0"
  reward_model_path: "/path/to/your/models/Skywork-Reward-V2-Llama-3.1-8B"
  k_samples: 8
```

### 4. 运行评估

使用您的配置文件执行主脚本。

```bash
cd ETrajEval
python run.py --config config.yaml
```

-   **生成阶段**: 脚本将根据 `prompt_file` 中的场景，在 `model1` 和 `model2` 之间运行聊天模拟。
-   **评分阶段**: 在生成之后（或使用已有的日志文件），此阶段将对每个对话中的情感轨迹进行评分。

### 5. 分析结果

结果将保存在 `output/` 目录中。

-   **聊天日志**: `output/chat_logs/{model1_vs_model2}/chat_logs_{language}.json`
    -   包含每个场景的完整对话记录。
-   **评估结果**: `output/eval_results/{model1_vs_model2}/eval_results_{language}.json`
    -   包含全面的结果，包括总体摘要分数、按心理学理论分类的表现，以及每次对话的详细逐轮分数。

## 🔬 核心概念解析

-   **轨迹度量指标**: ETrajEval的核心是三个共同描绘模型支持能力的度量指标。**BEL** 衡量整体的情感积极性，**ETV** 衡量模型推动积极变化的能力，而 **ECP** 则衡量这种变化的稳定性和方向。
-   **干扰事件**: 为了测试模型的韧性，我们在对话的特定轮次注入预定义的负面事件（例如，一封来自老板的批评邮件）。这模拟了现实世界的情感波动，并挑战模型提供适应性的支持。
-   **心理学基础**: 对话场景和模型约束是围绕既定的心理学原则设计的，如认知重评和情境选择，确保评估具有临床相关性。

## 🔧 如何添加你自己的模型

通过在 `config.yaml` 中添加新的模型定义，可以轻松集成新模型：

-   **对于本地Hugging Face模型**:
    ```yaml
    model_new:
      name: "My-Mistral-Finetune"
      type: "local"
      path_or_name: "/path/to/my/mistral/finetune"
      gpu_indices: [1]
      generation_params: { ... }
    ```
-   **对于基于API的模型**:
    ```yaml
    model_new:
      name: "Claude-3-Opus"
      type: "api"
      model_name: "claude-3-opus-20240229"
      api_config:
        api_key: "YOUR_ANTHROPIC_KEY"
        base_url: "https://api.anthropic.com/v1"
      generation_params: { ... }
    ```

<!-- ## 📄 引用

如果您在研究中使用了ETrajEval，请引用我们的论文：

```bibtex
@article{anonymous2025et,
  title={Detecting Emotional Dynamic Trajectories: An Evaluation Framework for Emotional Support in Language Models},
  author={Anonymous},
  journal={Details to be updated upon publication},
  year={2025}
}
``` -->  

## 🤝 贡献

我们欢迎任何形式的贡献！请随时提出Issue或提交Pull Request。

## 📝 开源协议

该项目采用 Apache 2.0 许可证。