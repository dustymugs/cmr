# Flags for CMR

To use a flag

```
python ... --FLAG_NAME=FLAG_VALUE ...
```

Example

```
python -m cmr.experiments.bird --name=bird_net --display_visuals --plot_scalars --data_dir=/cmr/CUB_200_2011/ --cache_dir=/cmr/cachedir/cub --num_epochs=500
```

## Training Flags

Extract flags with the following and then postprocess

```
cd /cmr
grep -R flags.DEFINE | egrep ^(data|experiments|nnutils).* | grep ":flags."
```

File | Flag | Default Value | Details
---- | ---- | ------------- | -------
data/base.py | data_dir | None | Data Directory for source images
data/base.py | cache_dir | None | Cache Directory for Annotations and SFM MAT files
data/base.py | img_size | 256 | image size
data/base.py | padding_frac | 0.05 |
data/base.py | jitter_frac | 0.05 |
data/base.py | split | "train" | [train, val, all, test], eval split
data/base.py | num_kps | 15 | The dataloader should override these.
data/base.py | n_data_workers | 4 | Number of data loading workers
experiments/shape.py | kp_loss_wt | 30. | keypoint loss weight
experiments/shape.py | mask_loss_wt | 2. | mask loss weight
experiments/shape.py | cam_loss_wt | 2. | weights to camera loss
experiments/shape.py | deform_reg_wt | 10. | reg to deformation
experiments/shape.py | triangle_reg_wt | 30. | weights to triangle smoothness prior
experiments/shape.py | vert2kp_loss_wt | .16 | reg to vertex assignment
experiments/shape.py | tex_loss_wt | .5 | weights to tex loss
experiments/shape.py | tex_dt_loss_wt | .5 | weights to tex dt loss
experiments/shape.py | use_gtpose | True | if true uses gt pose for projection, but camera still gets trained.
experiments/shape.py | include_weighted | False | if True, include weighted loss values to loss output
nnutils/train_utils.py | name | "exp_name" | Experiment Name
nnutils/train_utils.py | gpu_id | 0 | Which gpu to use
nnutils/train_utils.py | num_epochs | 1000 | Number of epochs to train
nnutils/train_utils.py | num_pretrain_epochs | 0 | If >0, we will pretain from an existing saved model.
nnutils/train_utils.py | learning_rate | 0.0001 | learning rate
nnutils/train_utils.py | beta1 | 0.9 | Momentum term of adam
nnutils/train_utils.py | use_sgd | False | if true uses sgd instead of adam, beta1 is used as mmomentu
nnutils/train_utils.py | batch_size | 16 | Size of minibatches
nnutils/train_utils.py | num_iter | 0 | Number of training iterations. 0 -> Use epoch_iter
nnutils/train_utils.py | checkpoint_dir | os.path.join(cache_path, snapshots) |
nnutils/train_utils.py | print_freq | 20 | scalar logging frequency
nnutils/train_utils.py | save_latest_freq | 10000 | save latest model every x iterations
nnutils/train_utils.py | save_epoch_freq | 50 | save model every k epochs
nnutils/train_utils.py | display_freq | 100 | visuals logging frequency
nnutils/train_utils.py | display_visuals | False | whether to display images
nnutils/train_utils.py | print_scalars | True | whether to print scalars
nnutils/train_utils.py | plot_scalars | False | whether to plot scalars
nnutils/train_utils.py | is_train | True | Are we training ?
nnutils/train_utils.py | display_id | 1 | Display Id
nnutils/train_utils.py | display_winsize | 256 | Display Size
nnutils/train_utils.py | display_port | 8097 | Display port
nnutils/train_utils.py | display_single_pane_ncols | 0 | if positive, display all images in a single visdom web panel with certain number of images per row.
nnutils/predictor.py | ignore_pred_delta_v | False | Use only mean shape for prediction
nnutils/predictor.py | use_sfm_ms | False | Uses sfm mean shape for prediction
nnutils/predictor.py | use_sfm_camera | False | Uses sfm mean camera
nnutils/mesh_net.py | symmetric | True | Use symmetric mesh or not
nnutils/mesh_net.py | nz_feat | 200 | Encoded feature size
nnutils/mesh_net.py | texture | True | if true uses texture!
nnutils/mesh_net.py | symmetric_texture | True | if true texture is symmetric!
nnutils/mesh_net.py | tex_size | 6 | Texture resolution per face
nnutils/mesh_net.py | subdivide | 3 | # to subdivide icosahedron, 3=642verts, 4=2562 verts
nnutils/mesh_net.py | use_deconv | False | If true uses Deconv
nnutils/mesh_net.py | upconv_mode | "bilinear" | upsample mode
nnutils/mesh_net.py | only_mean_sym | False | If true, only the mean shape is symmetric

