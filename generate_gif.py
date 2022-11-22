import glob
from PIL import Image


def make_gif(frame_folder):
    frames = list()
    for i in range(20):
        frames.append(Image.open(frame_folder+'/'+ str(i) + '.png'))
    frame_one = frames[0]
    frame_one.save("fishency_scene_0/eval_out/eval_out.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    make_gif("fishency_scene_0/eval_out")
