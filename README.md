# LLM Finetuner POC - For Finetuning and Evaluating LLM using SFT

## Inspiration

> This repo is inspired by [LLM Finetuning Hub](https://github.com/georgian-io/LLM-Finetuning-Hub) and [LAMA Factory](https://github.com/hiyouga/LLaMA-Factory)

## Changelog

[31/12/2023] We add the module of training the LLM that supports a lot of models

## Supported Training Approaches and Models

Currently we supported only Supervised Fine-Tuning in Full-Parameter, Partial-Parameter, LoRA, and QLoRA
The models supported as base model are [OpenLAMA 7B V2](openlm-research/open_llama_7b_v2)

> Use `--quantization_bit 4/8` argument to enable QLoRA.

## Provided Datasets

- For supervised fine-tuning:
  - [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
  - [GPT-4 Generated Data (en)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  - [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  - [LIMA (en)](https://huggingface.co/datasets/GAIR/lima)
  - [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)

Please refer to [data/README.md](data/README.md) for details.

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- Transformers, Datasets, Accelerate, PEFT and TRL
- sentencepiece, protobuf and tiktoken

## Getting Started

### Dependence Installation (optional)

```bash
git clone https://github.com/HazemMamdouh/LLM_Finetuner_POC.git
conda create -n LLMFinetuner python=3.10
conda activate LLMFinetuner
cd LLM_Finetuner_POC
pip install -r requirements.txt
```

### Train on a single GPU

#### Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

### Predict

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_predict \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

> We recommend using `--per_device_eval_batch_size=1` and `--max_target_length 128` at 4/8-bit predict.


This repository is licensed under the [Apache-2.0 License](LICENSE).


![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/LLaMA-Factory&type=Date)