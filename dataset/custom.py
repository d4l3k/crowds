import glob
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
import scipy.stats as st
from typing import List, NamedTuple, Tuple


def gkern(kernlen: int, nsig: float = 2.5) -> torch.Tensor:
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return torch.from_numpy(kern2d/kern2d.sum()).float()


def add_to(density, dist, y, x):
    h, w = dist.shape

    l = 0
    r = w
    b = 0
    t = h

    bottom = int(y-h/2)
    top = bottom + h
    left = int(x-w/2)
    right = left + w

    if bottom < 0:
        b += -bottom
        bottom = 0
    if top > density.shape[0]:
        t -= top-density.shape[0]
        top = density.shape[0]
    if left < 0:
        l += -left
        left = 0
    if right > density.shape[1]:
        r -= right-density.shape[1]
        right = density.shape[1]
    if b <= t and l <= r:
        density[bottom:top, left:right] += dist[b:t, l:r]


class Person(NamedTuple):
    leftx: int
    lefty: int
    rightx: int
    righty: int
    age: int
    gender: int


Pos = Tuple[int, int]


def iog_files() -> List[Tuple[str, Pos]]:
    files = []
    for file in glob.glob("iog_dataset/*/PersonData.txt"):
        dirname = os.path.dirname(file)
        with open(file) as f:
            poses = None
            pic = None
            for line in f.readlines():
                line = line.strip()
                if line.endswith(".jpg"):
                    if pic is not None:
                        src = os.path.join(dirname, pic)
                        files.append((src, poses))

                    pic = line
                    poses = []
                else:
                    p = Person(*map(int, line.split("\t")))
                    poses.append((
                        (p.leftx + p.rightx)/2,
                        (p.lefty+p.righty)/2
                    ))
    return files


assert len(iog_files())


def qrnf_files() -> List[Tuple[str, Pos]]:
    files = []
    for file in glob.glob('UCF-QNRF_ECCV18/*/*.jpg'):
        basepath = os.path.splitext(file)[0]
        metapath = basepath + "_ann.mat"
        meta = loadmat(metapath)
        files.append((file, meta['annPoints']))
    return files


assert len(qrnf_files())


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, train):
        self.files = iog_files() + qrnf_files()
        random.shuffle(self.files)
        num_train = int(len(self.files) * 0.95)
        if train:
            self.files = self.files[:num_train]
        else:
            self.files = self.files[num_train:]

        #self.files = qrnf_files()
        self.train = train
        self.augment = transforms.Compose([
            transforms.RandomGrayscale(),
        ])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.scale = (0.08, 1.0)
        self.ratio = (3. / 4., 4. / 3.)
        self.flip_chance = 0.5
        self.size = (480, 640)
        self.density_size = (116, 156)
        self.interpolation = Image.BILINEAR

    def __len__(self):
        return len(self.files)

    def _get_density(
        self, people: List[Pos], i: int, j: int, h: int, w: int,
        img_size: Tuple[int, int]
    ) -> torch.Tensor:
        num_people = len(people) * (h / img_size[0]) * (w / img_size[1])
        people_density = 0.5
        dot_size = np.mean(self.density_size) / np.sqrt(num_people) * people_density
        dot = gkern(dot_size)
        density = torch.zeros(self.density_size)
        for (x, y) in people:
            newx = int((x-j)/w*self.density_size[1])
            newy = int((y-i)/h*self.density_size[0])
            add_to(density, dot, newy, newx)

        return density.unsqueeze(dim=0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, people = self.files[index]
        img = Image.open(path).convert('RGB')
        if not self.train:
            i = 0
            j = 0
            h = img.height
            w = img.width
        else:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, self.scale, self.ratio)

        density = self._get_density(people, i, j, h, w, img.size)

        img = transforms.functional.resized_crop(
            img, i, j, h, w, self.size, self.interpolation)

        if self.train:
            if random.random() < self.flip_chance:
                img = transforms.functional.hflip(img)
                density = density.flip(dims=(2,))
            img = self.augment(img)

        img = self.transforms(img)

        return img, density


for train in [True, False]:
    dataset = CustomDataset(train)
    assert dataset[0]
