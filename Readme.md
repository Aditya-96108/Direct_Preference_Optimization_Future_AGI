Stepwise DPO Implementation & Experiment
Introduction
This repo implements Stepwise DPO for LLM long-chain reasoning, inspired by arXiv:2406.18629 and Lightman et al. (2023). Purposes: Mutual fit (technical preview/evaluation) and visible reasoning (via Git commits).

Scaled down due to limited resources: Tiny GSM8K subsets (10 train examples, ~2-15 pairs) and reduced sampling to fit free Colab GPU (time limits, disconnections). All tasks completed at small scale for proof-of-concept. AI assistance (Grok) noted in commits and LLM_USAGE.md.

Why Colab: Used for CPU/GPU-heavy tasks (model generation, training) as local hardware lacks sufficient VRAM/power; prevents crashes and enables quick prototyping without setup.

Why Colab Downloads: Downloaded models/data/results as zips to integrate into local VS Code project for GitHub, per user preference (no Google Drive linking); ensures reproducibility without cloud dependencies.

Approach
Data Generation: Rejection sampling on EleutherAI/gpt-neo-1.3B for step-wise pairs; wrong steps paired with correct continuations.
Reward Model: Phi-3-mini-4k-instruct (8-bit) as LLM evaluator for step correctness (yes/no probs).
Trainer: StepwiseDPOTrainer subclasses DPOTrainer; adds step-level loss aggregation (avg per batch/step).
Bonus: Fine-tune 1.3B model with LoRA; before/after eval on 10 tests.
Trade-offs: Small scale yields noisy gains; modular code for scaling. Next: Full datasets, larger models, advanced aggregation, hyperparam tuning.

Completed Tasks
LLM Reward Model: Phi-3 evaluates steps directly; tested successfully.
StepwiseDPOTrainer: Subclass with aggregation in get_batch_loss_metrics.
Bonus Improvement: Base acc 0.10 (1/10); Trained acc 0.20 (2/10); noisy but demonstrates method (results in results/results.txt).
Data Alt: Rejection sampling used.
Code: Python 3.10+, typed, PEP8. Reproducible via requirements.txt.

Tasks for Better Resources
Data Gen: Full GSM8K (7473 train); 10+ samples; A100 GPU.
Training: 3 epochs, batch=4; enable WandB.
Eval: Full test (1319); add metrics vs. baseline.
Run: pip install -r requirements.txt; python data_generation.py && train.py && evaluate.py. Configs in configs/.

Front-End Demo
demo.py (Gradio): Input math question; generates steps with live eval (correct/wrong highlights, scores); compares base/trained. Run: python demo.py (localhost:7860). Example: "2+2" → Color-coded trace, boxed answer.

Installation
Clone: git clone https://github.com/yourusername/yourrepo.git
Env: python -m venv env && source env/bin/activate
Install: pip install -r requirements.txt
