from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import math

def prepare_test_sem():
    test_sem_paths = sorted(glob('/data2/samsung/test/SEM/*.png'))
    return test_sem_paths

def prepare_result_sem():
    result_sem_paths = sorted(glob('/home/compu/project_samsung2/result/submission/*.png'))
    return result_sem_paths

def prepare_train_sem():
    train_sem_paths = sorted(glob('/data2/samsung/train/SEM/*/*/*.png'))
    return train_sem_paths

def prepare_train_sem_per_Depth():
    train_sem_paths_1 = sorted(glob(f'/data2/samsung/train/SEM/Depth_110/*/*.png'))
    train_sem_paths_2 = sorted(glob(f'/data2/samsung/train/SEM/Depth_120/*/*.png'))
    train_sem_paths_3 = sorted(glob(f'/data2/samsung/train/SEM/Depth_130/*/*.png'))
    train_sem_paths_4 = sorted(glob(f'/data2/samsung/train/SEM/Depth_140/*/*.png'))
    return train_sem_paths_1, train_sem_paths_2, train_sem_paths_3, train_sem_paths_4

def prepare_simulation_sem():
    simulation_sem_paths = sorted(glob('/data2/samsung/simulation_data/SEM/*/*/*.png'))
    return simulation_sem_paths

def prepare_simulation_sem_per_Case():
    simulation_sem_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_1/*/*.png'))
    simulation_sem_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_2/*/*.png'))
    simulation_sem_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_3/*/*.png'))
    simulation_sem_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_4/*/*.png'))
    return simulation_sem_paths_1, simulation_sem_paths_2, simulation_sem_paths_3, simulation_sem_paths_4

def prepare_simulation_sem_per_8x(Case):
    simulation_sem_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/80/*.png'))
    simulation_sem_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/81/*.png'))
    simulation_sem_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/82/*.png'))
    simulation_sem_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/83/*.png'))
    simulation_sem_paths_5 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/84/*.png'))
    return simulation_sem_paths_1, simulation_sem_paths_2, simulation_sem_paths_3, simulation_sem_paths_4, simulation_sem_paths_5

def prepare_simulation_depth():
    simulation_depth_paths = sorted(glob('/data2/samsung/simulation_data/Depth/*/*/*.png'))
    return simulation_depth_paths

def prepare_simulation_depth_per_Case():
    simulation_depth_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_1/*/*.png'))
    simulation_depth_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_2/*/*.png'))
    simulation_depth_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_3/*/*.png'))
    simulation_depth_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_4/*/*.png'))
    return simulation_depth_paths_1, simulation_depth_paths_2, simulation_depth_paths_3, simulation_depth_paths_4

def prepare_simulation_depth_per_8x(Case):
    simulation_depth_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/80/*.png'))
    simulation_depth_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/81/*.png'))
    simulation_depth_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/82/*.png'))
    simulation_depth_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/83/*.png'))
    simulation_depth_paths_5 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/84/*.png'))
    return simulation_depth_paths_1, simulation_depth_paths_2, simulation_depth_paths_3, simulation_depth_paths_4, simulation_depth_paths_5

def prepare_sem():
    simulation_sem_paths = sorted(glob('/data2/samsung/simulation_data/SEM/*/*/*.png'))

    train_sem_paths = sorted(glob('/data2/samsung/train/SEM/*/*/*.png'))
    test_sem_paths = sorted(glob('/data2/samsung/test/SEM/*.png'))

    return simulation_sem_paths, train_sem_paths, test_sem_paths

def load_imgs(img_paths):
    imgs = []

    for idx, img_path in tqdm(enumerate(img_paths)):
        img = cv2.imread(img_path, 0)
        imgs.append(img)

    imgs = np.array(imgs)

    return imgs

def calcHist(imgs):
    imgs = imgs.reshape(-1, imgs.shape[-1])

    hist = cv2.calcHist([imgs], [0], None, [256], [0, 256])
    hist = hist / imgs.size * 100
    info = dict(max=round(imgs.max(), 3),
                min=round(imgs.min(), 3),
                avg=round(imgs.mean(), 3),
                std=round(imgs.std(), 3))

    return hist, info
def drawHist_per_dataset(paths, prefix, mode='depth', imgs_min=0, imgs_max=0):
    imgs = load_imgs(paths)

    #imgs_max, imgs_min = imgs.max(), imgs.min()
    imgs = (imgs-imgs_min).astype(float) * 255 / (imgs_max-imgs_min)
    imgs = imgs.astype('uint8')
    hist_sem, hist_info = calcHist(imgs)

    if mode == 'depth':
        for i in range(256):
            if hist_sem[i] > 1.0:
                hist_sem[i] = 1.0
        max_value = 1.0
    else:
        max_value = hist_sem.max()

    list = [255, 174, 141, 115, 94, 75, 57, 42, 27, 13, 0]
    max_values = [max_value if idx in list else 0 for idx in range(256)]



    fig = plt.figure(figsize=(12, 5))
    spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[11, 1])

    ax0 = fig.add_subplot(spec[0])
    ax0.bar(np.arange(256), hist_sem.squeeze(), color='b')
    #ax0.bar(np.arange(256), max_values, color='red')
    ax0.set_xlim([0, 256])
    ax0.set_xticks([i*10 for i in range(26)])
    ax0.set_title(f'{prefix} Dataset Histogram')

    ax1 = fig.add_subplot(spec[1])
    ax1.text(0, 0.8, f"Max : {hist_info['max']}", fontsize=10)
    ax1.text(0, 0.6, f"Min : {hist_info['min']}", fontsize=10)
    ax1.text(0, 0.4, f"Avg : {hist_info['avg']}", fontsize=10)
    ax1.text(0, 0.2, f"Std : {hist_info['std']}", fontsize=10)
    ax1.axis('off')
    #plt.show()
    plt.savefig(f'/home/compu/project_samsung2/histogram/NormV2_{prefix}.png', dpi=300)

if __name__ == '__main__':
    simulation_sem = prepare_simulation_sem()
    drawHist_per_dataset(simulation_sem, 'simulation_sem', 'sem', 0, 212)

    simulation_depth = prepare_simulation_depth()
    drawHist_per_dataset(simulation_depth, 'simulation_depth', 'depth', 0, 170)

    simulation_sem_1, simulation_sem_2, simulation_sem_3, simulation_sem_4 = prepare_simulation_sem_per_Case()
    simulation_depth_1, simulation_depth_2, simulation_depth_3, simulation_depth_4 = prepare_simulation_depth_per_Case()
    drawHist_per_dataset(simulation_sem_1, 'simulation_sem_case 1', 'sem', 0, 212)
    drawHist_per_dataset(simulation_depth_1, 'simulation_depth_case 1', 'depth', 0, 170)
    drawHist_per_dataset(simulation_sem_2, 'simulation_sem_case 2', 'sem', 0, 212)
    drawHist_per_dataset(simulation_depth_2, 'simulation_depth_case 2', 'depth', 0, 170)
    drawHist_per_dataset(simulation_sem_3, 'simulation_sem_case 3', 'sem', 0, 212)
    drawHist_per_dataset(simulation_depth_3, 'simulation_depth_case 3', 'depth', 0, 170)
    drawHist_per_dataset(simulation_sem_4, 'simulation_sem_case 4', 'sem', 0, 212)
    drawHist_per_dataset(simulation_depth_4, 'simulation_depth_case 4', 'depth', 0, 170)

    train_sem = prepare_train_sem()
    drawHist_per_dataset(train_sem, 'train_sem', 'sem', 0, 212)

    train_sem_1, train_sem_2, train_sem_3, train_sem_4 = prepare_train_sem_per_Depth()
    drawHist_per_dataset(train_sem_1, 'train_sem_case 110', 'sem', 0, 212)
    drawHist_per_dataset(train_sem_2, 'train_sem_case 120', 'sem', 0, 212)
    drawHist_per_dataset(train_sem_3, 'train_sem_case 130', 'sem', 0, 212)
    drawHist_per_dataset(train_sem_4, 'train_sem_case 140', 'sem', 0, 212)

    test_sem = prepare_test_sem()
    drawHist_per_dataset(test_sem, 'test_sem', 'sem', 0, 212)

    result_sem = prepare_result_sem()
    drawHist_per_dataset(result_sem, 'result_sem', 'depth', 0, 170)

