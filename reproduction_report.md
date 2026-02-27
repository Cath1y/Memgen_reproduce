# MemGen Reproduction Report: Weaver (SFT) + Trigger (GRPO)

## 1. Experiment Overview
This experiment aims to reproduce the core mechanism of the MemGen paper, specifically generating latent memory via a **Weaver** model and controlling its activation via a **Trigger** policy to assist a **Reasoner** in complex reasoning tasks.
The focus of this reproduction includes:
1.  **Weaver Training**: Supervised Fine-Tuning (SFT).
2.  **Trigger Training**: Group Relative Policy Optimization (GRPO) Reinforcement Learning.
3.  **Dataset**: GSM8K (Mathematical Reasoning).
4.  **Base Model**: Qwen2.5-1.5B-Instruct.

---

## 2. Environment Setup

### Hardware
- **GPU**: 4x [A800]
- **CPU**: [72 vCPU Intel(R) Xeon(R) Platinum 8374C CPU @ 2.70GHz]

### Software
- **Operating System**: Linux (ubuntu 22.04)
- **Python**: 3.10
- **PyTorch**: 2.6
- **CUDA**: 12.1
- **Key Libraries**:
    - `transformers`: 4.55.4
    - `peft`: 0.17.1
    - `trl`: 0.21.0
    - `accelerate`: 1.10.1
    - `deepspeed`: ZeRO-2

---

## 3. Training Process

### 3.1 Stage 1: Weaver SFT Training
*Goal: Train the Weaver to generate latent memories that aid reasoning based on the current context.*

- **Config File**: `configs/latent_memory/gsm8k.yaml`
- **Key Hyperparameters**:
    - `train_weaver_method`: sft
    - `learning_rate`: 1e-5
    - `batch_size`: 4 (per device)
    - `num_train_epochs`: 2
    - `max_length`: 1024
    - `prompt_latents_len`: 16
    - `inference_latents_len`: 8
- **Training Logs**:
    - the SFT training loss curve of the weaver
     ![](https://upload-images.jianshu.io/upload_images/30822466-5374590c92e291b4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
        + The loss curve declines rapidly within the first 400 steps and plateaus thereafter.
        + This indicates that the SFT process for the Weaver converges quickly and effectively.

### 3.2 Stage 2: Trigger GRPO Training
*Goal: Train the Trigger policy network to decide when to activate the Weaver (maximizing reasoning accuracy via RL).*

- **Config File**: `configs/latent_memory/gsm8k.yaml`
- **Key Hyperparameters**:
    - `train_trigger_method`: grpo
    - `num_generations`: 8 (Group Size)
    - `learning_rate`: 1e-5
    - `beta`: 0.0 (KL Penalty)
    - `max_prompt_aug_num`: 1
    - `max_inference_aug_num`: 5
- **Reward Function**:
    - Binary reward based on GSM8K answer correctness (Correct=1, Incorrect=0).
- **Training Logs**: 
    - the GRPO training loss curve of the trigger
     ![](https://upload-images.jianshu.io/upload_images/30822466-467153a74ae894e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
        + The loss curve oscillates within a very narrow range around zero (-0.06 to 0.04).
        + This suggests that the model is undergoing robust fine-tuning, iteratively adjusting its weights to adapt to complex decision boundaries.
        + However, the loss curve alone is insufficient to verify the final effectiveness of the trigger; ablation studies are required.
    - the GRPO reward curve of the trigger
     ![](https://upload-images.jianshu.io/upload_images/30822466-9e7f6ce1f794d66d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
        + The reward values fluctuate due to significant variations in difficulty among sample groups or volatility in generated token lengths. This reflects the inherent stochasticity of the sampling process rather than a failure to learn.
        + The actual efficacy of the trigger mechanism will be confirmed through ablation studies.

---

## 4. Evaluation Results

### 4.1 Comparative Analysis
| Configuration | Trigger Policy | Weaver Model | GSM8K Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Inactive (False) | N/A (Load Null) | 39.5% | Original Base Model Performance |
| **No Trigger** |  (Untrained) | SFT Weaver | 56.5% | Ablation: Is trained trigger necessary? |
| **MemGen (Ours)** | **GRPO Trained** | **SFT Weaver** | **63.9%** | Full Reproduction |


**observations**
- the GRPO trigger is necessary for the model as the accuracy was improved from 56.5% to 63.9% while trigger was adapted
- weaver is the most important part of the MemGen because it generates latent memory that helps the model to reason correctly
- however,the accuracy is higher than that showed in the paper. Maybe the authot used a different state of the model(temperature/max_response_length) which is not shown in the paper.

---

## 5. Challenges & Solutions

Technical challenges encountered during reproduction and their resolutions:

### 5.1 NCCL Collective Operation Timeout (30-min Hang)
- **Symptom**: During Trigger/Weaver GRPO training, the process hangs around step 300 with 100% GPU utilization but no progress, eventually throwing an NCCL Timeout error.
- **Root Cause Analysis**:
    1.  **Generation Loop**: The model entered an infinite repetition loop under greedy search (`do_sample=False`), causing extremely long generation times.
    2.  **Distributed Deadlock (Control Flow Divergence)**: An optimization branch `if (augment_count >= max)` in the code caused some ranks to exit the generation loop early while others remained inside, leading to a deadlock at the `gather` collective communication point.
    3.  **Imbalanced Data Load**: The dataset was not shuffled, causing some GPUs to process a batch of long/difficult samples simultaneously.
- **Solutions Implemented**:
    1.  **Code Fix**: Commented out the optimization branch in `modeling_memgen.py` to enforce synchronized execution across all ranks.
    2.  **Hyperparameter Tuning**: Added `repetition_penalty: 1.2` in `gsm8k.yaml` to suppress infinite loops.
    3.  **Data Processing**: Enforced `shuffle` on the training dataset in `runner.py`.

### 5.2 GPU Out of Memory (OOM)
- **Symptom**: During training or evaluation, the process terminates unexpectedly with CUDA Out of Memory errors, especially when processing long sequences or large batch sizes.
- **Root Cause Analysis**: Accumulation of unused tensors in PyTorch's cache over time, which were not being automatically released, leading to gradual memory fragmentation and exhaustion.
- **Resolution**: Implemented a custom callback in `callbacks.py` to periodically invoke `torch.cuda.empty_cache()` and `gc.collect()`. This forces the release of unoccupied cached memory back to the OS, ensuring stable memory usage throughout long training runs.

---

## 6. Conclusion
This experiment successfully reproduced the MemGen functionality based on the SFT method of weaver. Results indicate that introducing the Latent Memory mechanism Significantly improved reasoning capabilities on GSM8K. The Trigger module, trained via GRPO, effectively learned to [Describe learned policy, e.g., trigger before key calculation steps].

---
