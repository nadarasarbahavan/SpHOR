
SEED=0

DATASET='aircraft'
AUG_M=15
AUG_N=2
LABEL_SMOOTHING=0.2



python -m methods.ARPL.osr_firststage --lr=0.1 --model='timm_resnet50_repl' --optim='sgd' \
                             --transform='rand-augment'  \
                            --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N} --loss='sphor' --label_smoothing=${LABEL_SMOOTHING} \
                            --dataset=${DATASET} --image_size=448 \
                            --scheduler='cosine_warm_restarts_warmup' --split_train_val='False' --batch_size=64 --num_workers=3 --max-epoch=600 \
                             --num_restarts=2 --seed=${SEED} --gpus 0 --feat_dim=2048  --projdim 1024 --notpretrained


