# Medical Q&A Assistant - QLoRA Fine-tuning + GRPO

Fine-tuned LLaMA-3.1-8B-Instruct on medical conversation data using QLoRA, then aligned it with GRPO to make it behave more like an actual helpful medical assistant rather than just a text completer.

## What's in the notebook

**Data** — using `lavita/ChatDoctor-HealthCareMagic-100k`, real anonymized patient-doctor conversations. Took 5k examples for SFT training and a separate 1k slice for GRPO.

**SFT** — standard QLoRA fine-tuning with 4-bit quantization (nf4) so it fits on a Colab T4. LoRA rank 8, targeting q/v projection layers. 3 epochs on the ChatDoctor data with the LLaMA chat template.

**GRPO** — loaded the merged SFT model, added fresh LoRA adapters, then ran GRPOTrainer with a rule-based reward function instead of a separate reward model. The reward function checks for things like appropriate response length, whether the model recommends seeing a doctor for serious symptoms, epistemic hedging, and penalizes specific drug dosage mentions hard.

## Stack

- `transformers` + `trl` + `peft` + `bitsandbytes`
- Colab T4 (free tier)
- Weights logged to wandb
