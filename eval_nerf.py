import torch
import json
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

import load_fishency
from nerf_components import *
from nerf_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_models(params, model_fname, fine_model_fname=None):
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
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(
            model_fname, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_fname))
    model.to(device)
    model_params = list(model.parameters())
    if params['use_fine_model']:
        fine_model = NeRF(encoder.d_output, n_layers=params['n_layers'], d_filter=params['d_filter'], skip=params['skip'],
                          d_viewdirs=d_viewdirs)
        if not torch.cuda.is_available():
            fine_model.load_state_dict(torch.load(
                fine_model_fname, map_location=torch.device('cpu')))
        else:
            fine_model.load_state_dict(torch.load(fine_model_fname))
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    return model, fine_model, encode, encode_viewdirs


def predict(pose, height, width, focal, near, far, model, fine_model, encode, encode_viewdirs, params):
    model.eval()
    fine_model.eval()
    rays_o, rays_d = get_rays(height, width, focal, pose)
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
    reshaped_out = rgb_predicted.reshape([height, width, 3])
    reshaped_out = torch.permute(reshaped_out, (2, 0, 1))

    return reshaped_out


def predict_many(poses, height, width, focal, near, far, model, fine_model, encode, encode_viewdirs, params):
    results = list()
    for i in tqdm(range(poses.shape[0])):
        prediction = predict(poses[i], height, width, focal, near,
                             far, model, fine_model, encode, encode_viewdirs, params)
        save_as_image(prediction, Path('fishency_scene_0/eval_out'), i)
        results.append(prediction)
    return results


def save_as_video(frames, fps, sec, save_dir):
    pass


def save_as_images(frames, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        save_image(frame, str(save_dir) + '/' + str(i) + '.png')


def save_as_image(frame, save_dir, i):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_image(frame, str(save_dir) + '/' + str(i) + '.png')


if __name__ == "__main__":
    experiment_path = Path(
        'training_outputs/lr0.0005_gamma0.5_batch_size256_date2022-11-18-12-25-26/')
    params_fname = experiment_path / 'params.json'
    model_fname = experiment_path / 'model_00099000.pth'
    fine_model_fname = experiment_path / 'fine_model_00099000.pth'
    test_poses_fname = 'fishency_scene_0/transformation_matrices.json'

    _, t_poses, t_focal, t_hwnf, _, _ = load_fishency.load(
        images_path="fishency_scene_0")

    with open(params_fname) as params_file:
        params = json.load(params_file)

    with open(test_poses_fname) as test_poses_file:
        test_poses_tmp = json.load(test_poses_file)

    max_pose = max(list(map(int, test_poses_tmp.keys())))
    test_poses = list()
    for i in range(max_pose + 1):
        test_poses.append(np.asarray(test_poses_tmp[str(i)], dtype=np.float32))

    stacked_test_poses = np.stack(test_poses, axis=0)
    stacked_test_poses = torch.from_numpy(stacked_test_poses).to(device)

    model, fine_model, encode, encode_viewdirs = init_models(
        params, model_fname, fine_model_fname)

    height, width, near, far = t_hwnf
    focal = t_focal

    frames = predict_many(stacked_test_poses, height, width, focal,
                          near, far, model, fine_model, encode, encode_viewdirs, params)

    # save_as_images(frames, Path('fishency_scene_0/eval_out'))
