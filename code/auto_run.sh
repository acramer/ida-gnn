. ~/miniconda3/etc/profile.d/conda.sh
conda activate ida

echo 'hello world' > runlog.txt
mv runlog.txt best_model/model.th

git add .
git commit -m "GCP test"
git push

# # LR ~ 3hr -> 3hr
# python main.py  -sdE 30 -B 128  -L 0.1    -A 'dla' --des dla_b128_1L1_sgd_plat
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --des dla_b128_2L1_sgd_plat
# python main.py  -sdE 30 -B 128  -L 0.05   -A 'dla' --des dla_b128_2L5_sgd_plat
# python main.py  -sdE 30 -B 128  -L 0.001  -A 'dla' --des dla_b128_3L1_sgd_plat
# python main.py  -sdE 30 -B 128  -L 0.005  -A 'dla' --des dla_b128_3L5_sgd_plat
# python main.py  -sdE 30 -B 128  -L 0.0001 -A 'dla' --des dla_b128_4L1_sgd_plat
# 
# # Batch ~ 2hr -> 2hr 10m
# python main.py  -sdE 30 -B 32   -L 0.01   -A 'dla' --des dla_b32_2L1_sgd_plat
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --des dla_b128_2L1_sgd_plat
# python main.py  -sdE 30 -B 256  -L 0.01   -A 'dla' --des dla_b256_2L1_sgd_plat
# python main.py  -sdE 30 -B 1024 -L 0.01   -A 'dla' --des dla_b1k_2L1_sgd_plat
# 
# # Adam/SGD and Plat/Cosine ~ 4hr
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --des dla_b128_2L1_sgd_plat
# python main.py -asdE 30 -B 128  -L 0.01   -A 'dla' --des dla_b128_2L1_adam_plat
# python main.py  -sdE 30 -B 128  -L 0.001  -A 'dla' --des dla_b128_3L1_sgd_plat
# python main.py -asdE 30 -B 128  -L 0.001  -A 'dla' --des dla_b128_3L1_adam_plat
# 
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --cosine --des dla_b128_2L1_sgd_cos
# python main.py -asdE 30 -B 128  -L 0.01   -A 'dla' --cosine --des dla_b128_2L1_adam_cos
# python main.py  -sdE 30 -B 128  -L 0.001  -A 'dla' --cosine --des dla_b128_3L1_sgd_cos
# python main.py -asdE 30 -B 128  -L 0.001  -A 'dla' --cosine --des dla_b128_3L1_adam_cos
# 
# # Affine, Crop, Simple, All 2.5 hrs
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --Aug simple --des dla_b128_2L1_sgd_plat_simp
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --Aug affine --des dla_b128_2L1_sgd_plat_aff
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --Aug crop   --des dla_b128_2L1_sgd_plat_crop
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --Aug rotate --des dla_b128_2L1_sgd_plat_rot
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --Aug all    --des dla_b128_2L1_sgd_plat_all
# 
# python main.py  -sdE 10 -B 128  -L 0.01   -A 'dla' --Aug normal        --des dla_b128_2L1_sgd_plat_norm
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --Aug 'light-heavy' --des dla_b128_2L1_sgd_plat_lihe
# python main.py  -sdE 30 -B 128  -L 0.01   -A 'dla' --Aug croprot       --des dla_b128_2L1_sgd_plat_cprt
#
# # Compare Swish to ReLU
# python main.py  -sdE 15 -B 128  -L 0.01   -A 'dla' --Act relu   --des dla_b128_2L1_sgd_cos_relu
# python main.py  -sdE 15 -B 128  -L 0.01   -A 'dla' --Act swish  --des dla_b128_2L1_sgd_cos_swish
# 
# # 256 LR/Optim ~ 50min
# python main.py  -sdE   5 -B 256  -L 0.01   -A 'dla' --cosine --des dla_b256_2L1_sgd_cos
# python main.py  -sdE   5 -B 256  -L 0.005  -A 'dla' --cosine --des dla_b256_3L5_sgd_cos
# python main.py  -sdE   5 -B 256  -L 0.001  -A 'dla' --cosine --des dla_b256_3L1_sgd_cos
# python main.py  -sdE   5 -B 256  -L 0.0005 -A 'dla' --cosine --des dla_b256_4L5_sgd_cos
# python main.py  -sdE   5 -B 256  -L 0.0001 -A 'dla' --cosine --des dla_b256_4L1_sgd_cos
# python main.py -asdE   5 -B 256  -L 0.01   -A 'dla'          --des dla_b256_2L1_adam_plat
# python main.py -asdE   5 -B 256  -L 0.005  -A 'dla'          --des dla_b256_3L5_adam_plat
# python main.py -asdE   5 -B 256  -L 0.001  -A 'dla'          --des dla_b256_3L1_adam_plat
# python main.py -asdE   5 -B 256  -L 0.0005 -A 'dla'          --des dla_b256_4L5_adam_plat
# python main.py -asdE   5 -B 256  -L 0.0001 -A 'dla'          --des dla_b256_4L1_adam_plat
#
# # Run 1 ~ 3.5hr
# python main.py  -sdE 200 -B 128  -L 0.01   -A 'dla' --Act swish  --des DLA_b128_2L1_sgd_cos_swish
#
# # Run 3
#python main.py -S 5 -asdE 200 -B 256  -L 0.0005  -A 'dla' -D 0.2 -W 0.0005 --Aug all --Act swish --des DLA_b256_5L4_adam_plat_swish_all
#
# # Run 4 ~ 4hr
# python main.py -S 5 -sdE 200 -B 256  -L 0.01  -A 'dla' -D 0.3 -W 0.0005 --Aug all --Act swish --cosine --des DLA-4
# 
# # Dropout search ~ 2 hr
# python main.py      -asdE 30  -B 256  -L 0.0005  -A 'dla' -D 0.1 -W 0.0005           --Act swish --des dla_b256_5L4_1D1_adam_plat_swish
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.2 -W 0.0005           --Act swish --des dla_b256_5L4_2D1_adam_plat_swish
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.3 -W 0.0005           --Act swish --des dla_b256_5L4_3D1_adam_plat_swish
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.5 -W 0.0005           --Act swish --des dla_b256_5L4_5D1_adam_plat_swish
# 
# # Weight Decay search ~ 2.5hr
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.2 -W 0.001            --Act swish --des dla_b256_5L4_2D1_2W3_adam_plat_swish
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.2 -W 0.0005           --Act swish --des dla_b256_5L4_2D1_5W4_adam_plat_swish
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.2 -W 0.0002           --Act swish --des dla_b256_5L4_2D1_2W4_adam_plat_swish
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.2 -W 0.0001           --Act swish --des dla_b256_5L4_2D1_1W4_adam_plat_swish
# python main.py      -asdE 20  -B 256  -L 0.0005  -A 'dla' -D 0.2 -W 0.00005          --Act swish --des dla_b256_5L4_2D1_1W5_adam_plat_swish
#
# # 32 Batch Test ~ 30min
# python main.py -S 5 -sdE 20  -B 32   -L 0.01  -A 'dla' -D 0.3 -W 0.0005 --Aug all --Act swish --cosine --des dla-32test
# 
# # Run 5 ~ 5hr
# python main.py -S 5  -sdE 200 -B 32   -L 0.01   -A 'dla' -D 0.3 -W 0.0005 --Aug all --Act swish --cosine --des DLA-5
# 
# # Run 6 ~ 5hr
# python main.py -S 5  -sdE 200 -B 256  -L 0.01   -A 'dla' -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 3 --cosine --des DLA-6
#                                                                                                               
# # Run 7 ~ 5hr                                                                                                 
# python main.py -S 5 -asdE 200 -B 32  -L 0.005  -A 'dla' -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 3 --cosine --des DLA-7

# 32 LR/Optim ~ 1hr
# # python main.py  -sdE   5 -B 32  -L 0.05   -A 'dla' --cosine --des dla_b32_2L5_sgd_cos
# # python main.py  -sdE   5 -B 32  -L 0.01   -A 'dla' --cosine --des dla_b32_2L1_sgd_cos
# # python main.py  -sdE   5 -B 32  -L 0.005  -A 'dla' --cosine --des dla_b32_3L5_sgd_cos
# # python main.py  -sdE   5 -B 32  -L 0.001  -A 'dla' --cosine --des dla_b32_3L1_sgd_cos
# # python main.py  -sdE   5 -B 32  -L 0.0005 -A 'dla' --cosine --des dla_b32_4L5_sgd_cos
# python main.py -asdE   5 -B 32  -L 0.05   -A 'dla'          --des dla_b32_2L5_adam_plat
# python main.py -asdE   5 -B 32  -L 0.01   -A 'dla'          --des dla_b32_2L1_adam_plat
# python main.py -asdE   5 -B 32  -L 0.005  -A 'dla'          --des dla_b32_3L5_adam_plat
# python main.py -asdE   5 -B 32  -L 0.001  -A 'dla'          --des dla_b32_3L1_adam_plat
# python main.py -asdE   5 -B 32  -L 0.0005 -A 'dla'          --des dla_b32_4L5_adam_plat
# python main.py -asdE   5 -B 32  -L 0.0001  -A 'dla'          --des dla_b32_4L1_adam_plat
# python main.py -asdE   5 -B 32  -L 0.00005 -A 'dla'          --des dla_b32_5L5_adam_plat
# python main.py -asdE   5 -B 32  -L 0.00001 -A 'dla'          --des dla_b32_5L1_adam_plat
# 
# # Run 8 ~ 5hr                                                                                                 
# python main.py -S 5 -asdE 200 -B 32  -L 0.0005  -A 'dla' -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 3 --cosine --des DLA-8
# 
# # Block Type Search ~ 1hr
# python main.py      -asdE 20  -B 128 -L 0.01   -A dla -T dblock  -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 2 --cosine --des dblock
# python main.py      -asdE 20  -B 128 -L 0.01   -A dla -T mbconv1 -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 2 --cosine --des mbconv1
# python main.py      -asdE 20  -B 128 -L 0.01   -A dla -T mbconv6 -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 2 --cosine --des mbconv6
# 
# # Run 9 ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T dblock  -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 2          --des DLA_9
# 
# # Run 1 clone ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T dblock                             --Act swish --dla_depth 2          --des DLA_1_Clone
# 
# # Run 10 ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T mbconv1  -D 0.3 -W 0.0005 --Aug all --Act swish --dla_depth 2          --des DLA_10
# 
# # Run 11 ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T mbconv6                             --Act swish --dla_depth 2          --des DLA_11

# python main.py --mode predict --load models/120-DLA_1_Clone2/modeli100.th
# cp predictions.npy preds/predictions_112_i100_94.npy
# python main.py --mode predict --load models/120-DLA_1_Clone2/modeli15.th
# cp predictions.npy preds/predictions_112_i15_89.npy
# python main.py --mode predict --load models/120-DLA_1_Clone2/modeli5.th
# cp predictions.npy preds/predictions_112_i5_81.npy
# 
# # python main.py --mode predict --load models/120-DLA_1_Clone2/modeli100.th --predict 'test/public_train_images.npy'
# # cp predictions.npy preds/train_112_i100_94.npy
# # python main.py --mode predict --load models/120-DLA_1_Clone2/modeli15.th --predict 'test/public_train_images.npy'
# # cp predictions.npy preds/train_112_i15_89.npy
# # python main.py --mode predict --load models/120-DLA_1_Clone2/modeli5.th --predict 'test/public_train_images.npy'
# # cp predictions.npy preds/train_112_i5_81.npy
# # 
# # python main.py --mode predict --load models/120-DLA_1_Clone2/modeli100.th --predict 'test/public_test_images.npy'
# # cp predictions.npy preds/test_112_i100_94.npy
# # python main.py --mode predict --load models/120-DLA_1_Clone2/modeli15.th --predict 'test/public_test_images.npy'
# # cp predictions.npy preds/test_112_i15_89.npy
# # python main.py --mode predict --load models/120-DLA_1_Clone2/modeli5.th --predict 'test/public_train_images.npy'
# # cp predictions.npy preds/train_112_i5_81.npy
# 
# git add .
# git commit -m 'preds added'
# git push

# # Run 12 ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T mbconv1                             --Act swish --dla_depth 2          --des dla_mb1_drop0
# 
# # Run 13 ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T mbconv1  -D 0.3                     --Act swish --dla_depth 2          --des dla_mb1_drop3

# # Run 1 clone ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T dblock                             --Act swish --dla_depth 2          --des DLA_1_Clone2

# # Run 14 ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T mbconv1                             --Act swish --dla_depth 3          --des dla_mb1_depth3
# 
# python main.py --mode predict --load models/121-dla_mb1_depth3/model.th
# cp predictions.npy preds/predictions_121.npy
# 
# git add .
# git commit -m 'preds added'
# git push
# 
# # Run 14 ~ 5hr
# python main.py -S 5  -sdE 200 -B 128 -L 0.01   -A dla -T mbconv1   -D 0.1                    --Act swish --dla_depth 3          --des dla_mb1_depth3_drop1
# 
# python main.py --mode predict --load models/122-dla_mb1_depth3_drop1/modeli170.th
# cp predictions.npy preds/predictions_122.npy
# 
# git add .
# git commit -m 'preds added'
# git push


# # Run 14 ~ 5hr
# python main.py -S 5  -sedE 200 -B 128 -L 0.01   -A dla -T dblock                              --Act swish --dla_depth 2          --des dla_sqeeze_excite

# python main.py --mode predict --load models/124-dla_sqeeze_excite/modeli100.th
# cp predictions.npy preds/predictions_124.npy
# 
# git add .
# git commit -m 'preds added'
# git push
# 
# python main.py -S 5  -sadE 100 -B 128 -L 0.001   -A efficient                                  --Act swish                       --des ENet
# 
# python main.py --mode predict --load models/125-ENet/model.th
# cp predictions.npy preds/predictions_126.npy
# 
# git add .
# git commit -m 'preds added'
# git push

# python main.py -S 5  -sedE 100 -B 128 -L 0.01   -A dla -T dblock  -D 0.1                       --Act swish --dla_depth 2          --des dla_drop_sqeeze_excite
# python main.py -S 5  -sedE 100 -B 128 -L 0.01   -A dla -T dblock  -D 0.1           --Aug all   --Act swish --dla_depth 2          --des dla_daug_se


# # (0) DLA-34 with All Data Aug
# python main.py -S 5 -sdE 150 -B 128 -L 0.01 -A dla --Act swish       --Aug all --des dla_Adaug
# 
# # (1) DLA-SE-34 with All Data Aug
# python main.py -S 5 -sdE 150 -B 128 -L 0.01 -A dla --Act swish    -e --Aug all --des dla_Adaug_se_nodrop
# 
# # (2) DLA-SE-34 with All Data Aug
# python main.py -S 5 -sdE 150 -B 128 -L 0.01 -A dla --Act swish    -eD 0.05 --Aug all --des dla_se_D05
# python main.py -S 5 -sdE 150 -B 128 -L 0.01 -A dla --Act swish    -eD 0.2  --Aug all --des dla_se_D2
# python main.py -S 5 -sdE 150 -B 128 -L 0.01 -A dla --Act swish    -eD 0.3  --Aug all --des dla_se_D3
# python main.py -S 5 -sdE 150 -B 128 -L 0.01 -A dla --Act swish    -eD 0.5  --Aug all --des dla_se_D5



# FINAL
# python main.py --mode predict --load models/134-dla_se_D2/model.th
# cp predictions.npy preds/predictions_134.npy
# python main.py --mode predict --load models/135-dla_se_D3/model.th
# cp predictions.npy preds/predictions_135.npy
# python main.py --mode predict --load models/136-dla_se_D5/model.th
# cp predictions.npy preds/predictions_136.npy
# 
# ACC=$(python main.py --mode test --load models/135-dla_se_D3/model.th)
# 
# 
# cp models/models/135-dla_se_D3/model.th ../saved_models/model.th
# cp models/models/135-dla_se_D3/model.th best_model/model.th
# 
# git add .
# git commit -m "Forgot to add Model"
# git push


# # Instructions
# # All recently trained models will be saved in models folder under the largest numbered model
# #     - the -S 0 flag and argument specifies that the model should be saved at the end of training
# #     - the -S N flag and argument specifies that the model should be saved every N epochs,
# #            into the same folder, with the name modeli<E>.th, where <E> is the epoch number
# #     - to resume training on a model, simply specify model.th with --load <path/to/model.th> flag and argument
# # Training DLA-34
# python main.py -S 0 -nsdE 120 -B 128 -L 0.01 -A dla --Act swish
# # Training Best Model (DLA-SE-34 + ADA + Dropout rate 0.3)
# python main.py -S 0 -nsdE 150 -B 128 -L 0.01 -A dla --Act swish -eD 0.3  --Aug all

