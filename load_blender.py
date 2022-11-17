import numpy as np
import matplotlib.pyplot as plt


def load(data_path='tiny_nerf_data.npz', n_training=100, testimg_idx=101):
    data = np.load(data_path)
    images, poses = data['images'], data['poses']
    focal = data['focal']

    height, width = images.shape[1:3]
    near, far = 2., 6.
    hwnf = (height, width, near, far)

    testimg, testpose = images[testimg_idx], poses[testimg_idx]

    return images, poses, focal, hwnf, testimg, testpose


if __name__ == "__main__":

    images, poses, focal, hwnf, testimg, testpose = load()

    print(f'Images shape: {images.shape}')
    print(f'Poses shape: {poses.shape}')
    print(f'Focal length: {focal}')
    print(hwnf)

    print(images.dtype)
    print(poses[0].dtype)
    print(focal.dtype)


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
