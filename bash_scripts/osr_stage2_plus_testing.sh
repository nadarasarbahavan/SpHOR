
SEED=0
DATASET='aircraft'

#Maintain the original RAND-AUGMENT parameters as Stage 01
AUG_M=15
AUG_N=2

#Update this with the relevant EXP_ID from the Stage 01 
EXP_ID='update_with_relevant_exp_id_from_stage_01'
EXP_ID_EPOCH='600'

python -m methods.ARPL.osr_classifier --lr=0.01 --model='timm_resnet50_repl' \
                             --transform='rand-augment' --optim=sgd  --scheduler='cosine'\
                            --rand_aug_m=15 --rand_aug_n=2 --loss='Softmax' --label_smoothing=0 \
                            --dataset='aircraft' --image_size=448 \
                            --split_train_val='False' --batch_size=128 --num_workers=3 --max-epoch=20 --max_epoch_stageone="${EXP_ID_EPOCH}" \
                             --num_restarts=0 --seed=0 --gpus 0 --feat_dim=2048 --exp_id="${EXP_ID}"  

