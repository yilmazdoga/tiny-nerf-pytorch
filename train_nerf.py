import torch
import json
import time
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import load_blender
import load_fishency
from nerf_components import *
from nerf_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_models(params):
    r"""
    Initialize models, encoders, and optimizer for NeRF training.
    """
    # Encoders
    encoder = PositionalEncoder(
        params['d_input'], params['n_freqs'], log_space=params['log_space'])

    def encode(x): return encoder(x)

    # View direction encoders
    if params['use_viewdirs']:
        encoder_viewdirs = PositionalEncoder(
            params['d_input'], params['n_freqs_views'], log_space=params['log_space'])

        def encode_viewdirs(x): return encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(encoder.d_output, n_layers=params['n_layers'], d_filter=params['d_filter'], skip=params['skip'],
                 d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if params['use_fine_model']:
        fine_model = NeRF(encoder.d_output, n_layers=params['n_layers'], d_filter=params['d_filter'], skip=params['skip'],
                          d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=params['lr'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=params['milestones'], gamma=params['gamma'])

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=100)

    return model, fine_model, encode, encode_viewdirs, optimizer, scheduler, warmup_stopper


def train(images, poses, focal, near, far, model, fine_model, encode, encode_viewdirs, optimizer, scheduler, warmup_stopper, params):
    """
    Launch training session for NeRF.
    """

    training_name = str(params['experiment_name']) + \
        "@" + str(time.strftime("%Y-%m-%d-%H-%M-%S"))

    training_save_dir = Path(params['save_dir']) / training_name
    training_save_dir.mkdir(parents=True, exist_ok=True)

    with open(training_save_dir / "params.json", "w") as outfile:
        json.dump(params, outfile)

    writer = SummaryWriter(training_save_dir)

    # Shuffle rays across all images.
    if not params['one_image_per_step']:
        height, width = images.shape[1:3]
        all_rays = torch.stack(
            [torch.stack(get_rays(height, width, focal, p), 0) for p in poses], 0)
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in range(params['n_iters']):
        model.train()
        if fine_model != None:
            fine_model.train()

        if params['one_image_per_step']:
            # Randomly pick an image as the target.
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)
            if params['center_crop'] and i < params['center_crop_iters']:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        else:
            # Random over all images.
            batch = rays_rgb[i_batch:i_batch + params['batch_size']]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += params['batch_size']
            # Shuffle after one epoch
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape([-1, 3])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = nerf_forward(rays_o, rays_d,
                               near, far, encode, model,
                               kwargs_sample_stratified=params['kwargs_sample_stratified'],
                               n_samples_hierarchical=params['n_samples_hierarchical'],
                               kwargs_sample_hierarchical=params['kwargs_sample_hierarchical'],
                               fine_model=fine_model,
                               viewdirs_encoding_fn=encode_viewdirs,
                               chunksize=params['chunksize'])

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
        rgb_predicted = outputs['rgb_map']
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        writer.add_scalar("Loss/train_mse", loss, i)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute mean-squared error between predicted and target images.
        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())
        writer.add_scalar("Loss/train_psnr", psnr, i)

        # Evaluate testimg at given display rate.
        if i % params['display_rate'] == 0:
            model.eval()
            if fine_model != None:
                fine_model.eval()
            height, width = testimg.shape[:2]
            rays_o, rays_d = get_rays(height, width, focal, testpose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(rays_o, rays_d,
                                   near, far, encode, model,
                                   kwargs_sample_stratified=params['kwargs_sample_stratified'],
                                   n_samples_hierarchical=params['n_samples_hierarchical'],
                                   kwargs_sample_hierarchical=params['kwargs_sample_hierarchical'],
                                   fine_model=fine_model,
                                   viewdirs_encoding_fn=encode_viewdirs,
                                   chunksize=params['chunksize'])

            rgb_predicted = outputs['rgb_map']
            loss = torch.nn.functional.mse_loss(
                rgb_predicted, testimg.reshape(-1, 3))
            writer.add_scalar("Loss/eval_mse", loss, i)
            val_psnr = -10. * torch.log10(loss)
            writer.add_scalar("Loss/eval_psnr", val_psnr, i)
            val_psnrs.append(val_psnr.item())
            iternums.append(i)
            reshaped_out = rgb_predicted.reshape([height, width, 3])
            reshaped_out = torch.permute(reshaped_out, (2, 0, 1))
            writer.add_image('TestPred', reshaped_out, i)

        if i % params['save_rate'] == 0:
            # Save current models
            model_name = "model_" + str(i).zfill(8) + ".pth"
            fine_model_name = "fine_model_" + str(i).zfill(8) + ".pth"
            torch.save(model.state_dict(), training_save_dir / model_name)
            if params['use_fine_model']:
                torch.save(fine_model.state_dict(),
                           training_save_dir / fine_model_name)

        # Check PSNR for issues and stop if any are found.
        if i == params['warmup_iters'] - 1:
            if val_psnr < params['warmup_min_fitness']:
                print(
                    f'Val PSNR {val_psnr} below warmup_min_fitness. Stopping...')
                return False, train_psnrs, val_psnrs
        elif i < params['warmup_iters']:
            if warmup_stopper is not None and warmup_stopper(i, psnr):
                print(
                    f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
                return False, train_psnrs, val_psnrs

        scheduler.step()
    return True, train_psnrs, val_psnrs


if __name__ == "__main__":

    params = {'d_input': 3,                     # Number of input dimensions
              'experiment_name': 'nerf_experiment_blender',
              'scene_name': 'fishency_scene_0',
              'n_freqs': 10,                    # Number of encoding functions for samples
              'log_space': True,                # If set, frequencies scale in log space
              'use_viewdirs': True,             # If set, use view direction as input
              'n_freqs_views': 4,               # Number of encoding functions for views
              'n_samples': 64,                  # Number of spatial samples per ray
              'perturb': True,                  # If set, applies noise to sample positions
              'inverse_depth': False,           # If set, samples points linearly in inverse depth
              'd_filter': 128,                  # Dimensions of linear layer filters
              'n_layers': 2,                    # Number of layers in network bottleneck
              'skip': [],                       # Layers at which to apply input residual
              'use_fine_model': True,           # If set, creates a fine model
              'd_filter_fine': 128,             # Dimensions of linear layer filters of fine network
              'n_layers_fine': 6,               # Number of layers in fine network bottleneck
              'save_dir': "training_outputs",
              'save_rate': 1000,
              'n_samples_hierarchical': 64,     # Number of samples per ray
              'perturb_hierarchical': False,    # If set, applies noise to sample positions
              'lr': 5e-4,
              'milestones': [20000, 50000, 90000],
              'gamma': 0.5,
              'n_iters': 100000,
              'batch_size': 2**8,
              'one_image_per_step': False,
              'chunksize': 2**8,                # Modify as needed to fit in GPU memory
              # Crop the center of image (one_image_per_)
              'center_crop': True,
              'center_crop_iters': 50,          # Stop cropping center after this many epochs
              'display_rate': 25,               # Display test output every X epochs
              'warmup_iters': 200,              # Number of iterations during warmup phase
              'warmup_min_fitness': 10.0,       # Min val PSNR to continue training at warmup_iters
              'n_restarts': 10,                 # Number of times to restart if training stalls
              'kwargs_sample_stratified': {'n_samples': 64, 'perturb': True, 'inverse_depth': False},
              'kwargs_sample_hierarchical': {'perturb': True}
              }

    images, poses, focal, hwnf, testimg, testpose = load_blender.load()

    height, width, near, far = hwnf

    images = torch.from_numpy(images).to(device)
    poses = torch.from_numpy(poses).to(device)
    focal = torch.from_numpy(focal).to(device)
    testimg = torch.from_numpy(testimg).to(device)
    testpose = torch.from_numpy(testpose).to(device)

    for _ in range(params['n_restarts']):
        model, fine_model, encode, encode_viewdirs, optimizer, scheduler, warmup_stopper = init_models(
            params)
        success, train_psnrs, val_psnrs = train(
            images, poses, focal, near, far, model, fine_model, encode, encode_viewdirs, optimizer, scheduler, warmup_stopper, params)
        if success and val_psnrs[-1] >= params['warmup_min_fitness']:
            print('Training successful!')
            break

    print('')
    print(f'Done!')
