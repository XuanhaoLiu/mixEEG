# mixEEG (CogSci 25)
🔥 Our paper **mixEEG: Enhancing EEG Federated Learning for Cross-subject EEG Classification with Tailored mixup** has officially been accepted as CogSci 25 for Oral presentation (13.8%).

![mixeeg](figure.png)

## How to run our code

At first, you should creat two dirs called 'logs' and 'checkpoints', then you can cd into 'src', and run mixEEG:

1. --model is the model type, it can be mlp or cnn.

2. --dataset is the dataset, it can be seed or chbmit.
   
4. --frac is the fraction for updating local model, it can be 0.1 to 1.

5. --iid is Independent and identically distributed, in our paper is always 1.

6. --epochs is the global updating round.

7. --lr is the learning rate for the local model.

8. --mixup_strategy is the strategy for mixup, it can be "none" for the vanilla FedAvg, "lin" for linear mixup, "cha" for channel mixup, and "fre" for frequency mixup.

9. --mixup_alpha is useful only under the linear mixup mode.

10. --mixup_subtype is the subtype of each mixup strategy, the meanings are the same as the paper.

11. --gpu is the GPU id.

**the DG FL settings:**
```shell
python3 federated_eeg_LOSO.py --model=mlp --dataset=seed --target_id=15 --num_users=14 --frac=0.2 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="lin" --mixup_alpha=1.0 --mixup_subtype="none"
python3 federated_eeg_LOSO.py --model=mlp --dataset=seed --target_id=15 --num_users=14 --frac=0.2 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="lin" --mixup_alpha=0.2 --mixup_subtype="none" --gpu=1
python3 federated_eeg_LOSO.py --model=mlp --dataset=seed --target_id=15 --num_users=14 --frac=0.2 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="cha" --mixup_alpha=0.2 --mixup_subtype="random" --gpu=1
python3 federated_eeg_LOSO.py --model=mlp --dataset=seed --target_id=15 --num_users=14 --frac=0.2 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="cha" --mixup_alpha=0.2 --mixup_subtype="binary" --gpu=2
python3 federated_eeg_LOSO.py --model=mlp --dataset=seed --target_id=15 --num_users=14 --frac=0.2 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="fre" --mixup_alpha=0.2 --mixup_subtype="cut" --gpu=0
python3 federated_eeg_LOSO.py --model=mlp --dataset=seed --target_id=15 --num_users=14 --frac=0.2 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="fre" --mixup_alpha=0.2 --mixup_subtype="cross" --gpu=0
python3 federated_eeg_LOSO.py --model=mlp --dataset=seed --target_id=15 --num_users=14 --frac=0.2 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="none" --mixup_alpha=0.2 --mixup_subtype="cross" --gpu=0
python3 federated_eeg_LOSO.py --model=cnn --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="none" --mixup_alpha=0.2 --mixup_subtype="cross" --gpu=1
python3 federated_eeg_LOSO.py --model=cnn --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="lin" --mixup_alpha=0.2 --mixup_subtype="cross" --gpu=1
python3 federated_eeg_LOSO.py --model=cnn --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="lin" --mixup_alpha=5.0 --mixup_subtype="cross" --gpu=1
python3 federated_eeg_LOSO.py --model=cnn --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="cha" --mixup_alpha=0.2 --mixup_subtype="random" --gpu=1
python3 federated_eeg_LOSO.py --model=cnn --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="cha" --mixup_alpha=0.2 --mixup_subtype="binary" --gpu=1
python3 federated_eeg_LOSO.py --model=cnn --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="fre" --mixup_alpha=0.2 --mixup_subtype="cut" --gpu=2
python3 federated_eeg_LOSO.py --model=cnn --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=100 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="fre" --mixup_alpha=0.2 --mixup_subtype="cross" --gpu=2
```

**the DA FL settings:** only need to replace federated_eeg_LOSO.py to federated_eeg_LOSO_DA.py
```shell
python3 federated_eeg_LOSO_DA.py --model=mlp --dataset=chbmit --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=10 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="none" --mixup_alpha=0.2 --mixup_subtype="cross" --gpu=1
python3 federated_eeg_LOSO_DA.py --model=cnn --dataset=seed --target_id=10 --num_users=9 --frac=0.3 --iid=1 --epochs=10 --lr=0.01 --local_ep=5 --local_bs=32 --mixup_strategy="none" --mixup_alpha=0.2 --mixup_subtype="cross" --gpu=1
```

## BibTeX
```
@inproceedings{liu2025mixeeg,
  title={mix{EEG}: Enhancing {EEG} Federated Learning for Cross-subject {EEG} Classification with Tailored mixup},
  author={Liu, Xuan-Hao and Lu, Bao-Liang and Zheng, Wei-Long},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  year={2025}
}
```

## Acknowledgement
This code is modified based on the FedAvg code from [This repo](https://github.com/zj-jayzhang/FedAvg).
