
This is the official implementation of our paper [A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers](https://arxiv.org/abs/2509.19947), accepted by NeurIPS 2025. 

## Requirements
conda create --name GeneralComponents python=3.8
conda activate GeneralComponents
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.22.3
pip install jupyter_core==5.8.1
pip install opencv-python==4.5.5.64
pip install scipy==1.10.1

## A Quick Start
**Step 1: Calculate metrics for sample selection**

```
CUDA_VISIBLE_DEVICES=0 python cal_metric.py --output_dir save_metric
```

**Step 2: Train backdoored model with different sample selection methods**

```
CUDA_VISIBLE_DEVICES=0 python train_backdoor.py --output_dir save_metric --result_dir save_res --selection forget --backdoor_type badnets --y_target 0 --select_epoch 10
```

`--select epoch` specifies the epoch used to calculate the statistics. `--output_dir` must be the same with the one used in cal_metric.py

## Citing
If this work or our codes are useful for your research, please kindly cite our paper as follows.

```
@misc{wu2025setgeneralizedcomponentsachieve,
      title={A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers}, 
      author={Zhixiao Wu and Yao Lu and Jie Wen and Hao Sun and Qi Zhou and Guangming Lu},
      year={2025},
      eprint={2509.19947},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
}
```
