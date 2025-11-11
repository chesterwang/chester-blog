---
title: "Huggingface 类库学习"
date: 2025-08-22
lastmod: 2025-08-22
tags:
  - LLM
keywords:
  - LLM
  - HuggingFace
description: "Huggingface 类库学习"
---




## Hugging Face Hub


1. Models, Spaces, and Datasets are hosted on the Hugging Face Hub as [Git repositories](https://git-scm.com/about)
2. Do you have files larger than 10MB? Those files should be tracked with `git-lfs`, which you can initialize with: `git lfs install`
3. Note that if your files are larger than **5GB** you’ll also need to run: `hf lfs-enable-largefiles .`
4. Pull Request 之所以叫这个名字，是因为它准确地描述了请求者（你）和被请求者（项目维护者）之间的动作和方向。
    1. PR = “我改好了，请你把我这边的修改拉（pull）进你的主分支吧。”这个 “pull” 并不是指你自己去拉，而是 **请求项目维护者去拉你的代码**。
5. [Templates](https://huggingface.co/docs/transformers/en/chat_templating)


## transformers

1. quickstart
    1. load a pretrained model
    2. run inference with [Pipeline](https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/pipelines#transformers.Pipeline)
    3. fine-tune a model with [Trainer](https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/trainer#transformers.Trainer)
2. [Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto)
    1. Instantiating one of [AutoConfig](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoConfig), [AutoModel](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoModel), and [AutoTokenizer](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/auto#transformers.AutoTokenizer) will directly create a class of the relevant architecture.
3. transformers库中 AutoImageProcessor 实例话出来的processor 的作用都有哪些
    1. 一旦你通过 `AutoImageProcessor.from_pretrained()` 实例化了一个处理器，这个 **processor 实例** 就成为了一个强大的工具，专门用于准备图像数据，使其能够被特定的预训练视觉模型使用。它的主要作用可以概括为以下几点：
    2. 图像标准化 (Normalization)
    3. 图像尺寸调整 (Resizing) 和裁剪 (Cropping)
    4. 通道格式转换 (Channel Format Conversion)
    5. 图像到张量转换 (Image to Tensor Conversion)
    6. 批处理 (Batching)
    7. 数据增强 (Data Augmentation)
4. transoformer库中 TFAutoModel和AutoModel的区别是什么
    1. **`AutoModel`**：用于加载 **PyTorch** 框架下的模型。
    2. **`TFAutoModel`**：用于加载 **TensorFlow 2.0** 框架下的模型。
    3. **`FlaxAutoModel`**: 用于加载 **Flax**（基于 JAX 的框架）下的模型。
5. AutoModel 类的后缀
    1. LM
        1. CausalLM
        2. MaskedLM
        3. MaskedGeneration
        4. sequenceClassification
        5. TokenClassification
        6. NextSentencePrediction
        7. MultipleChoice
        8. Seq2SeqLM
        9. QuestionAnswering
6. Backbone
    1. A backbone is a model used for feature extraction for higher level computer vision tasks such as object detection and image classification.
7. Data Collator
    1. Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of the same type as the elements of `train_dataset` or `eval_dataset`.
    2. 我理解 Data Collator 就是把多个列放在一起，就是collator的字面意思，但其实内部还是会有一些具体的数据处理逻辑，比如padding、数据扩增等
8. pipeline
    1. `from transformers import pipeline`
    2. `generator = pipeline(model="openai-community/gpt2")`
    3. `generator("I can't believe you did such a ", do_sample=False)`
    4. `[{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]`
9. chat with models
    1. Templates

## Diffusers


1. LoRA
    1. Add a LoRA to a pipeline with the [load_lora_weights()](https://huggingface.co/docs/diffusers/v0.35.1/en/api/loaders/lora#diffusers.loaders.QwenImageLoraLoaderMixin.load_lora_weights) method. Some LoRA’s require a special word to trigger it, such as `Realism`, in the example below. Check a LoRA’s model card to see if it requires a trigger word.
    2. LoRA文件就是一种插件

## Sentence Transformers

sentence embedding demo
```python
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```


```

```


---
## trl



1. [Quickstart](https://huggingface.co/docs/trl/quickstart) 简单的helloworld程序。
    1. **evaluation**: The important thing is that this process should yield a scalar value for each query/response pair.
    2. 这里的例子代码应该已经过时了，PPOTrainer现在没有step这个方法。
2. [Dataset formats and types](https://huggingface.co/docs/trl/dataset_formats)
3. How-to guides
    1. customozing the training
        1. Memory efficient fine-tuning by sharing layers

#### PPO


1. PPO策略中的一些基础知识
    1. 深度学习硬件配置中的概念 device rank world_size node 分别是指什么
        1. 这四个概念的关系可以概括为：**node** 是物理机器。 一个 **node** 可以包含多个 **device**，device通常是指GPU。 每个 **device** 通常由一个独立的 **rank** 进程来控制。 所有 **rank** 进程的总数就是 **world_size**。
    2. PPO示例
        1. [trl/examples/scripts/ppo/ppo.py at v0.21.0 · huggingface/trl](https://github.com/huggingface/trl/blob/v0.21.0/examples/scripts/ppo/ppo.py)
        2. model=policy, ref_model=ref_policy, reward_model=reward_model, value_model=value_model,、
        3. reward_model其实是每一小步决策的即时的、直接奖励，value_model对每一小步决策的全局性、长期性后果进行预测。
        4. advantage 我直观理解是因为Q(s,a) 这个state的随机选择导致后续的所有reward都比较异常，所以要消除一部分state的随机性，所以减去state的value；如果训练数据中的Q(s,a)中的s足够多，那么就可以直接使用Q，但因为不够多有了随机性，所以Q代表的return分布就偏离了所谓的全局性的Q的分布，所以通过减去这个s的随机性从而拉回正常分布。
    3. Mixin 类本身不能独立实例化，它不是用来创建对象的。相反，它像一个“功能包”或“能力插件”，专门设计用来被其他类**继承**。当一个类继承了一个或多个 Mixin 类时，它就自动获得了这些 Mixin 类所定义的所有方法和属性。
    4. PPO中的数据处理
        1. `transformers.tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus` 方法中包含了 tokenizer中的padding处理。
        2. [PPO官方例子中](https://github.com/huggingface/trl/blob/v0.21.0/examples/scripts/ppo/ppo.py)中的代码，
            1. 为什么 padding_side 和后面的padding=false ，这两者不冲突么
                1. 代码中两个地方使用了tokenizer，
                    1. `PPOTrainer( args=training_args, processing_class=tokenizer,...)` 这里的tokenizer定义中是带有padding="left"的。
                        1. 这里的tokenizer 在PPOTrainer中是作为 processing_class 的作用。
                        2. 在PPOTrainer中 会根据 processing_class 生成一个 DataCollatorWithPadding 类用于数据进行处理，而该类的计算逻辑是使用tokenizer对数据仅仅进行padding操作（这里的padding是右对齐），这里不包括encoding操作。因为训练数据已经根据第二条中的 prepare_dataset进行了encoding。
                    2. prepare_dataset 中的 tokenize函数。
                        1. 该函数的输入是一条数据，所以不需要进行padding。
                    3. 总结： 即encodding和padding是分在两个地方进行处理的，分别是PPOTrainer实例化之前 和 PPOTrainer的train方法内部 分别进行encoding和padding。
                2. 至于为什么这样做，从代码中没有看出原因。可能是为了更细粒度的控制训练和评估各自场景下的逻辑。
    5. `trl.trainer.utils.selective_log_softmax`
        1.  其中实现针对FP32 FP64 采用了高效的 logsumexp方法，针对其他的格式采用了低效的方法，因为低精度格式计算容易出现问题。
        2. `logsumexp` 的标准定义是： $$\log \sum_{i} \exp(x_i)$$，如果直接按这个公式实现，当 $x_i$ 的值很大时，$\exp(x_i)$ 可能会导致数值**上溢（overflow）**，超出浮点数的表示范围，结果变为无穷大。
        3.   工程上的稳定实现方法
            1. 为了避免上溢，`logsumexp` 在大多数深度学习框架（如 PyTorch、TensorFlow）中都有一个经过优化的、数值稳定的实现。核心思想是利用对数函数的性质，将指数运算中的大数相加问题，转换为对数运算中的小范围数相加问题。
            2. 具体方法如下：
                1. **找到最大值**：首先，找到输入向量 $x$ 中的最大值 $x_{max}$。
                2. **转换公式**：利用 $e^{a+b} = e^a e^b$ 的性质，将原公式进行等价转换： $$\log \sum_{i} \exp(x_i) = \log \left( \exp(x_{max}) \sum_{i} \exp(x_i - x_{max}) \right)$$
                3. **拆分对数**：利用 $\log(ab) = \log(a) + \log(b)$ 的性质，进一步拆分： $$= \log(\exp(x_{max})) + \log \left( \sum_{i} \exp(x_i - x_{max}) \right)$$ $$= x_{max} + \log \left( \sum_{i} \exp(x_i - x_{max}) \right)$$
                4. 为什么这个方法更稳定？
                    1. **防止上溢**：在转换后的公式中，$x_i - x_{max}$ 的值都是**负数或者零**。这意味着 $\exp(x_i - x_{max})$ 的值都在 $(0, 1]$ 范围内。这样，即使 $x_i$ 非常大，指数运算的结果也不会上溢。
                    2. **保持精度**：虽然 $x_i - x_{max}$ 的值是负数，但它们之间的相对大小关系保持不变，这保证了计算结果的精确度。
                    3. **计算效率**：这个稳定的实现只需要额外进行一次 `max` 运算和一次加法运算，对整体计算效率影响很小。
                    4. 例如，在 PyTorch 中，`torch.logsumexp` 函数就是以这种方式实现的。当你使用它时，框架会自动处理这些数值稳定性的细节。
                    5. 这个工程优化方法在深度学习中非常重要，特别是在处理诸如 Softmax 交叉熵损失、信念传播（Belief Propagation）等需要大量指数和对数运算的场景。
2. **PPO策略的流程(from chatgpt)** ^PPO-process
    1. **一、PPO算法核心思想**
        1. PPO 属于**策略梯度（Policy Gradient）**家族，目标是通过不断优化策略参数，使得智能体在环境中获得更高的期望回报。  
        2. 它的关键在于：**在更新策略时限制更新幅度**，防止策略改变太大导致训练不稳定。
    2. **二、PPO的基本流程（典型版本：PPO-Clip）**
        1. **采样（Rollout）**  
              1. 使用当前策略 $\pi_{\theta_{old}}(a_t|s_t)$，与环境交互，收集一批数据： $(s_t, a_t, r_t, s_{t+1})$
              2. 并计算折扣回报 $R_t$ 。
        2. **计算优势函数（Advantage Estimate）**  
              1. 通常用 **GAE（Generalized Advantage Estimation）**：   $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$
              2. 其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
        3. **计算重要性比率（ratio）**  
              1. $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$    表示新旧策略在同一动作上的“概率变化”。
        4. **构建PPO的目标函数（Clipped Surrogate Objective）**  
              1. $\large L^{CLIP}(\theta) = \mathbb{E}_t \big[  \min(  r_t(\theta) \hat{A}_t,  \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t  )\big]$
              2. 当  $r_t(\theta)$  偏离 1 太多（超过 ±ε）时，会被截断（clip），防止过度更新。 ε 通常取 0.1～0.2。
        5. **优化目标 + 值函数 + 熵正则**  
              1. 实际优化的目标函数通常是三项之和：   $L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta]$  
              2. $L^{VF}$：值函数的MSE损失
              3. $S[\pi_\theta]$：策略的熵（鼓励探索）
        6. **多次小步更新（K epochs）**  
              1. 用同一批采样数据，在每个 mini-batch 上优化 K 轮。
        7. **更新旧策略参数**  
              1. $\theta_{old} \leftarrow \theta$

> **PPOTrainer中代码的流程总结** ^PPO-code-process
> 1. **外循环**：总训练batch数
>     1. 对 policy model 生成的每一个序列样本，输入到reward model ，序列中最后一个token的 hidden state 被输入到一个线性头，最终输出一个标量值，表示序列二分类的打分score。
>     2. 对 policy model 生成的每一个序列样本，每一个token输出前的vocabulary 分布和 ref_model的分布 两者计算kl散度（使用了近似方法使得计算加速），即输出一个 kl散度序列，序列中每个元素表示对应位置token的kl散度。
>     3. 将序列级的score 加到kl序列中每一个位置，结果即为 rewards 序列
>     4. 依次计算广义优势估计 gae，`GAE(t)=rewards(t) + gamma*value(t+1) - value(t)   + gamma*lam*GAE(t+1)`，这里lam是指lambda
>     5. 计算 return `return(t) =  GAE(t)+ value(t)` 。（这里把value理解成是 t步的状态state下，未来奖励的总和期望，即 状态本身的价值）。
>     6. value即 state value function，GAE即 针对的是action。
>     7. **内循环**：针对上述步骤生成的序列样本和对应的return、GAE、概率logp 序列，将样本拆分成 micro batch，执行下列步骤
>         1. value 优化的损失函数为 `(return - value(t))^2`，其实就是约束 优势advantage GAE 本身尽量。
>         2. 使用当前的policy model 计算action 的新概率值 $\large \log p_{\theta}$
>         3. policy gradient损失为 $\Large - GAE * exp(\log p_{\theta}-\log p_{\theta_{old}})$ 我想这里这么写是为了数值稳定性，因为一般是让模型输出 logp，然后计算 概率比，那么就直接相减然后取指数。背后其实就是 logsumexp算子。
>         4. 两个损失加载一起进行反向传播，即会修改当前的policy_model和value_model
> 2. 输出和保存各种模型

> **local_rollout_forward_batch_size**
> 是每个节点，在本地进行多批次的训练，每个批次的大小即为 local_rollout_forward_batch_size。

> **policy_model和ref_model在rollout中调用的方法不同**
> 3. policy_model使用trl.trainer.utils.batch_generation（是对所有计算节点的并行批量化计算）。就是在query之后拼接预测出来的response。
>     1. batch_generation 使用generation_config参数，代码中规定 max_new_tokens=args.response_length，即硬性指定了response长度。
>     2. response在生成出来之后，会合并所有的立马进行右padding。
>     3. 其中调用了GenerationMixin的generate方法，该方法的输出为token ids的序列。
> 4. ref_model 使用 forward。就是将整个query_response拼接结果全部输入到ref_model，一次性得出错位的预测结果。所以对logits索引的时候会往前错一位，并且最后一个位置是不需要使用的。
>     1. 我想batch_generation是需要进行复杂padding的，从而可能导致每一个小batch生成的输出长度是不一致的。但我想不到为什么这么调用的理由。
> 5. 总体上PPO有两个地方牵涉到概率的对比
>     1. 第一个是 policy_model和ref_model两个模型的KL散度，放置policy_model跑太远过于离谱。
>     2. 第二个是 policy_model和old_policy_model的样本概率对比，但不需要保留old_policy_old这个模型，因为训练中仅仅是使用了 old_policy_model的样本和其样本概率，所以通过old_policy_model一次采样出来一大堆数据之后（其中保留了样本概率 $\Large p_{theta_{old}}$）， old_policy_model就可以丢弃，仅仅通过使用采样的样本和样本概率来进行policy_model的训练优化。 那么就相当于是两个模型合二为一。 


> **class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):**
> The Qwen2 Model transformer with a sequence classification head on top (linear layer).  
> [`Qwen2ForSequenceClassification`] uses the **last token** in order to do the classification, as other causal models  (e.g. GPT-2) do.
> 计算的时候会找到 response中的 last_non_pad_token，输出对应的logitis，然后经过score方法 进行linear_unit 计算。

> **class AlbertForSequenceClassification(AlbertPreTrainedModel):**
> Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the **pooled  output**) e.g. for GLUE tasks.

> `dataclass` 注解是 Python 3.7 及以上引入的一个装饰器，作用是**简化类的编写，让类自动获得一些常用方法**（如 `__init__`, `__repr__`, `__eq__` 等），用于表示数据结构。

> 为什么 计算ref_logprob 使用 selective_log_softmax方法，按理说KL散度应该是对vocabulary的所有词进行KL散度计算啊？
> **理论上**，KL散度的定义是： $$K L \left(\right. p \parallel q \left.\right) = \underset{i}{\sum} p \left(\right. i \left.\right) log ⁡ \frac{p \left(\right. i \left.\right)}{q \left(\right. i \left.\right)}$$ 这里 (i) 是**整个 vocabulary** 的所有 token。
> **实际工程实现（RLHF/PPO场景）**：
> 6. 我们通常只关心模型实际“走出的路径”，即生成的 token 序列。
> 7. PPO/Reward Modeling 里，KL项是用来约束新模型（policy）不要偏离旧模型（reference/policy）的行为，**只需要对“已采样的 token”上的概率分布做约束**。


> `rewards[[actual_start, actual_end]] += scores`
> 赋值操作，rewards中存放的是kl散度，这里则是将kl散度对应最后一个token位置向右错一位的位置加上一个最终的reward score。
> 这里的actual_start是样本的index，actual_end是指每一个样本的最后一个token位置+1。

> **在 Hugging Face Transformers 中，类名前的 `Auto` 表示自动选择模型的意思。**
> 8. `AutoModelForSequenceClassification` 不是一个具体的模型类，而是一个**工厂类**。
> 9. 它可以根据你加载的 checkpoint（如 `"bert-base-uncased"`、`"roberta-base"`、自定义路径等），**自动实例化对应的具体模型类**（如 `BertForSequenceClassification`、`RobertaForSequenceClassification` 等）。
> 10. 这让你不需要关心底层是哪个模型，只要传入模型名或路径，它会自动帮你选择正确的模型实现。

> PPO中 value_model一般使用的是 sequence_classification model，比如 [示例脚本](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py) 中代码如下，value_model和reward_model使用的是同一个模型类的不同实例。

```python
value_model = AutoModelForSequenceClassification.from_pretrained(
    training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
)
```

这里举例 `transformers.models.qwen2.modeling_qwen2.Qwen2ForSequenceClassification`

> PPOTrainer中调用get_reward的时候，使用的是 sequence_classfication_model的 base_model_prefix指向的底层LLM原始模型  、 score方法（score方法就是在baseLLM模型后添加一个线性层，以映射到logits）。其效果就是对一对问答字符串序列输出一串reward数值。


> get_reward方法中的具体逻辑。
> lm_backbone底层使用的是基模型，其中 output_hidden_states 表示要把模型中所有层的hidden_states全部进行输出。所以在调用score的时候会只取最后一层的hidden_states，然后输入到score方法中（即再经过一层线性层得到logits）。效果就是对于问答字符串序列中的**每一个token**都会得到一个reward_logits数值。

```python
output = lm_backbone(  
    input_ids=input_ids,  
    attention_mask=attention_mask,  
    position_ids=position_ids,  
    return_dict=True,  
    output_hidden_states=True,  
    use_cache=False,  # otherwise mistral-based RM would error out  
)
reward_logits = model.score(output.hidden_states[-1])
# 这里的output.hidden_states 表示base_model的所有层的输出
# output.hidden_states[-1] 则表示最后一层的输出
# Qwen2ForSequenceClassification 的forward中计算 loss，就是按照最后一个token的最后一层输出 +  真实label 一起计算出交叉熵

return (  
    reward_logits,  
    reward_logits[  
        torch.arange(reward_logits.size(0), device=reward_logits.device),  
        sequence_lengths,  
    ].squeeze(-1),  
    sequence_lengths,  
)
# 这里表示返回 response序列中最后一个合法token的logits输出
```

`transformers.models.qwen2.modeling_qwen2.Qwen2ForSequenceClassification` 是一个例子模型，该模型的forward方法中计算了每一个token输出的分类的logits，然后仅仅获取了每一个序列的最后一个 non_padding_token的logits作为输出。

--- 

reward_model得到的一个序列一个reward值，但其实是最后一步的immediate reward。
value_model得到的每一个动作（即token）一个value值（即长期效果的评估的指标）。

---

**PPOTrainer中 为什么需要有 missing_eos_penalty？**

在 `PPOTrainer` 中存在 `missing_eos_penalty`，主要是为了解决 **生成结果没有包含终止符（如 `eos_token_id`）的情况**，防止模型生成不完整或异常的响应。
- 在文本生成任务中，模型通常会在响应结尾生成一个“终止符”（例如 `eos_token_id`），表示响应结束。
- 如果模型没有生成终止符，响应可能：
    - 超长（一直生成到最大长度）
    - 不完整（缺少语法上的结尾）
    - 影响后续评估和训练（如奖励模型、PPO等）
- 如果响应没有终止符，可能是模型没学会“何时结束”，这种响应一般是不符合任务预期的，**需要惩罚**。
- `missing_eos_penalty` 就是对这种情况加一个负分，鼓励模型在合适的时候生成终止符。

---

> [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

> PPOTrainer中为什么要whiten_rewards
> 在 `PPOTrainer` 中，存在一个参数 `whiten_rewards`，其作用是对奖励（reward）进行归一化/标准化（whitening）。
> 奖励的尺度和分布直接影响优势的分布，而优势分布又影响梯度更新的稳定性和训练速度。
> 如果奖励很大或很小，会导致策略梯度很大/很小，影响收敛速度，甚至导致梯度爆炸/消失。

> 那么在采样阶段计算的reward、 value等数据，因为在训练阶段这些数据中的很多会重新生成，所以采样阶段很多计算是不是浪费了?
> - **采样阶段的 value/logprob**只用一次，训练阶段会重新算“新策略”的 value/logprob。

---

```python
for t in reversed(range(gen_length)):  
    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0  
    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]  
    lastgaelam = delta + args.gamma * args.lam * lastgaelam  
    advantages_reversed.append(lastgaelam)  
advantages = torch.stack(advantages_reversed[::-1], axis=1)  
returns = advantages + values  
advantages = masked_whiten(advantages, ~padding_mask)  
advantages = torch.masked_fill(advantages, padding_mask, 0)
```
这里计算return和advantage。return就是 state value的一个真实采样，用来计算 value_loss，即return和估计的value之间的平方损失。advantage 就是 优势，即当前时刻以及每个之后的时刻所计算的advantage的打折之和，用来给策略梯度加权。

---

损失函数
```python
logprobs_diff = new_logprobs - mb_logprobs  
ratio = torch.exp(logprobs_diff)  
pg_losses = -mb_advantage * ratio  
pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)  
pg_loss_max = torch.max(pg_losses, pg_losses2)  
pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])  
loss = pg_loss + args.vf_coef * vf_loss
```
这块代码中为什么没有使用 logp的导数，原因是  $\Large \mathbb{E}_{\beta} \left[ \frac{\pi_{\theta}(a|s)}{\beta(a|s)} Q^{\pi}(s,a)\nabla_{\theta} \ln \pi_{\theta}(a|s) \right]$ ，其实就是 $\Large \mathbb{E}_{\beta} \left[ \frac{\nabla_{\theta} \pi_{\theta}(a|s)}{\beta(a|s)} Q^{\pi}(s,a) \right]$，那么对应的损失函数就是以$\large \theta$为优化参数的  $\Large - \mathbb{E}_{\beta} \left[ \frac{\pi_{\theta}(a|s)}{\beta(a|s)} Q^{\pi}(s,a) \right]$。


--- 

mb_logprobs 在loss中不会反向传播梯度么
因为 mb_logprobs计算的时候是在 `with torch.no_grad():` 中。

---

`vf_losses1 = torch.square(vpred - mb_return)` 这里的mb_return 就是后续多步累积的reward值。

---


为什么policy loss 和 value function loss 要加起来进行optimize？
1. 如果你设计了**分离的 policy model 和 value model**（即 Actor 和 Critic 完全分离），那么确实可以分开 optimize、分开 backward。
2. 但最主流的实现（比如 Huggingface Transformers 的 PPOTrainer）是**合一模型**，一个模型里有两个输出 head，参数是共享的，所以必须把 loss 合在一起，统一 backward 和 optimize。
3. 优化器（optimizer）会对所有参数做梯度更新，policy head 和 value head的梯度会分别回传到主干和各自 head。

---

在train 方法内，`ref_policy` 是自始至终 一致保持不变的。`ref_policy `只是避免 `policy_model` 跑得太远。而样本重用是使用importantce weight来解决的，即 $\large \mathbb{E}_{\beta} \left[ \frac{ \pi_{\theta}(a|s)}{\beta(a|s)} \right]$  。

---

**这个PPO算法的实现，为什么既有kl散度计算作为 reward，同时又有clip操作**

- **clip objective**：控制单次更新不要太大，防止训练不稳定。
- **KL 散度奖励**：保证整体策略不会逐渐偏离参考模型（human-preference-aligned policy）太远。
- KL 散度相当于在 **奖励层面**惩罚策略远离参考模型，避免模型跑偏；这就是所谓的“KL 奖励塑形”。
- clip 解决 **短期训练稳定性**；
- KL 奖励解决 **长期偏移问题**，相当于给模型加了个“牵引绳”。

你看到的实现里，**KL 散度不是直接当约束用，而是作为奖励项参与回报计算**；而 **clip 则在优化目标里约束策略更新**。这两者是互补关系。


---
---
#### DPO


TRPO中的PO和 DPO中的PO指的不是一个东西？

- **TRPO**：它的目标是最大化**环境奖励**。**TRPO** (Trust Region **Policy Optimization**)
- **DPO**：它的目标是最大化**人类偏好**。 **DPO** (Direct Preference **Optimization**)


---

DPO的直觉化理解
1. 整体的目标函数为 $$\large L(\theta) = \max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta} [r(x, y)] - \beta \text{KL}[\pi_\theta(y|x), \pi_{\theta_{\text{old}}}(y|x)]$$，函数表达的意义是，在一个奖励结构上，policy的结构必须尽量与奖励结构保持一致，但同时不要偏离老的policy太远，后者可以认为是一定程度的正则化。 
2. **形象化理解**，对于 y变量的概率分布和奖励分布（即横轴是y 的各种取值，纵轴是对应y值行为的概率值大小、奖励大小），那么用图形理解就是两条曲线（只不过一个是归一化的正值，一个是可能正负值均存在）。对于上述目标函数来说，暂且认为奖励分布$r(x,y)$就是真实的训练数据值（但其实是从 正负例response的 contrastive loss拟合出来的值），那么$L(\theta)$中的唯一变量就是 策略函数 $\pi_{\theta}$ ，这个时候目标函数$L(\theta)$其实就是一个以 $\pi_\theta$为自变量的一个函数（当然自变量本身就是概率分布函数，那么这其实就是一个泛函优化问题）；对上述目标函数进行变换，可以得出 $L(\theta)$取得最大值对应的自变量 $\pi_\theta$ 是有一个固定公式的，即 $$\large \pi_r(y|x) = \pi^*(y|x) = \frac{1}{Z(x)}\pi_{\theta_{\text{old}}}(y|x) \exp\left(\frac{1}{\beta}r(x,y)\right)$$。
    1. 这里可以极简的方法推导处理，如下 $$L =\mathbb{E}[r]-KL = \sum_{p} p*(r-\log(\frac{p}{q})) = \sum_{p} p*(\log(e^r)-\log(\frac{p}{q}))=\sum_{p} p*(\log(\frac{e^r*q}{p})) = - \sum_{p} p*(\log(\frac{e^r*q}{p})) = - KL(p, e^r*q)$$，而最后一个式子就是KL散度的公式取负数，KL散度公式存在最小值，那么$L$就存在最大值，最优点即为 $\large p = e^r*q$，最后公式外围套个归一化就是最终的推理结果。
3. 直觉化的理解就是  策略函数 -> 和奖励结构对应的最优策略函数 -> 得到最优的整体目标值。而奖励结构必须与真实的正负样例偏好结构一致，所以整体上就是 奖励结构和策略函数是绑死的有固定函数关系，而奖励结构通过损失函数与真实正负样例偏好对齐，那么反向来说就是 真实正负样例偏好->指导奖励结构计算 ->指导对应的策略函数， 而对应策略函数其实本来就应该达到一种极端策略（即使得奖励最大的response的概率直接拉到最大值1），但因为正则化的存在使得 策略函数是介于 老策略函数和 极端策略 之间的中间策略。
4. 假如没有正则化，那么整个过程就是极端策略，让正样例的生成概率无限大，负样例的生成概率无限小，也就是对应的logits之差无限大，那么正样例的概率直接为1，负样例的概率直接为0，但这样其实就是过拟合。所以需要正则化来限制，那么就是通过 用概率的logits来表示 $r(x,y)$，同时假定损失结构是 让 $r(x,y)$ 和 KL散度直接相加，来作为最终的目标函数。

直观上理解DPO的公式，即让正偏好的对应的策略action概率越大好，让负偏好对应的策略action概率越小越好。
$$\large L(\theta)=-\mathbb{E}_{(x,y^+,y^-) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}(y^+|x)}{\pi_{\theta_{old}}(y^+|x)}-\beta \log \frac{\pi_{\theta}(y^-|x)}{\pi_{\theta_{old}}(y^-|x)}\right)\right].$$
其中$\large \hat{r}_\theta(x,y) = \beta \log \left( \frac{\pi_\theta(y|x)}{\pi_{\theta_{old}}(y|x)} \right)$ 意思是，如果当前策略对应的预估奖励是多少。

---

直觉上我将DPO比作是一个三个铁杆上分别套着一个环，三个铁环之间 有两个绳子连接，传统的policy gradient是将通过拉最下面一个环，让上面两个间接连带着移动。而DPO是将上面两个环固化成一体，只要移动最下面一个，就能达到直接移动最上面铁环的目的。

DPO算法的训练标准数据，都是配对的。

DPO算法中损失函数中的logp指的是不是整条episode中答案的概率log，即在给定prompt的情况下，给出response的每个token概率的乘积，也就是logp的和。

--- 


## GRPO

1. DeepSeekMath 论文的解读 [[2025Q3-论文学习日记#2025-07-16 DeepSeekMath]]
2. GRPO 中使用了process supervision 信号，即单个reponse的多步推理，每一步都有标注的reward。然后 advantage 就是 归一化后的reward 在t步之后的累计和。即基于同一个prompt的所有response的reward的均值和方差 进行标准化  $\Large \tilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$   ，然后累计和 $\Large \hat{A}_{i,t} = \sum_{index(j) \geq t} \tilde{r}_i^{index(j)}$ ，这里t就是每一个token，j是每一个reasoning step，即每一个token其reward就是该token之后所有对应于 reasoning step end token的token的奖励之和。我想这里之所以没有使用$\gamma$ 可能是因为reasoning step数量本来都是极其有限的，并且采样应该会控制其数量。
3. 所以GRPO是需要使用 process reward model 对response的每一个推理步骤进行打分的。

---

The training data of the reward model is based on the rule judgment. 
Data Source: question in SFT dataset with outputs sampled from SFT model. Reward Function:
Rule (whether the answer is correct or not)
论文里的公式里有很多讲究。

---

原始论文中的公式19

$\large \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\mathbf{q} \sim P_{\text{sft}}(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)}   \large \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t} - \beta \left( \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1 \right) \right].$

$\large w_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x, y_{i,<t})}$

## GSPO

[GSPO Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)

$\large J_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim D, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G}\sum_{i=1}^G \min\left(s_i(\theta)\hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right) \right]$

$\large \hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_j)\}_{j=1}^G)}{\text{std}(\{r(x, y_j)\}_{j=1}^G)}$

$\large s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right)^{\frac{1}{|y_i|}} = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)$


**GSPO和GRPO强化学习算法的区别**

|       | GRPO                | GSPO                    |
| ----- | ------------------- | ----------------------- |
| 训练数据  | process supervision | outcome supervision     |
| 重要性采样 | 基于每一个token          | 基于整个序列，所有token重要性的的几何平均 |
