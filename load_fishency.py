import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path


def load(images_path='fishency_scene_0'):

    images_path = Path(images_path)
    images = list()
    poses = list()

    images.append(imread(images_path/"cam0.png"))
    images.append(imread(images_path/"cam1.png"))
    images.append(imread(images_path/"cam2.png"))
    images.append(imread(images_path/"cam3.png"))

    poses.append(np.array([[1., 2., 3., 4.],
                           [5., 6., 7., 8.],
                           [9., 10., 11., 12.],
                           [0., 0., 0., 1.]]))

    poses.append(np.array([[1., 2., 3., 4.],
                           [5., 6., 7., 8.],
                           [9., 10., 11., 12.],
                           [0., 0., 0., 1.]]))

    poses.append(np.array([[1., 2., 3., 4.],
                           [5., 6., 7., 8.],
                           [9., 10., 11., 12.],
                           [0., 0., 0., 1.]]))

    poses.append(np.array([[1., 2., 3., 4.],
                           [5., 6., 7., 8.],
                           [9., 10., 11., 12.],
                           [0., 0., 0., 1.]]))

    stacked_images = np.stack(images, axis=0)
    stacked_poses = np.stack(poses, axis=0)

    height, width = stacked_images.shape[1:3]
    near, far = 2., 6.

    focal = 0
    hwnf = (height, width, near, far)
    testimg = stacked_images[0]
    testpose = stacked_poses[0]

    return stacked_images, stacked_poses, focal, hwnf, testimg, testpose


if __name__ == "__main__":

    images, poses, focal, hwnf, testimg, testpose = load()

    print(f'Images shape: {images.shape}')
    print(f'Poses shape: {poses.shape}')
    print(f'Focal length: {focal}')
    print(hwnf)

    plt.imshow(testimg)
    print('Pose')
    print(testpose)

    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1)
                    for pose in poses])
    origins = poses[:, :3, -1]

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
        origins[..., 0].flatten(),
        origins[..., 1].flatten(),
        origins[..., 2].flatten(),
        dirs[..., 0].flatten(),
        dirs[..., 1].flatten(),
        dirs[..., 2].flatten(), length=0.5, normalize=True)
    plt.show()