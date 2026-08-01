import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data.dataset import SYSUDataset
from data.dataset import RegDBDataset
from data.dataset import LLCMDataset

from data.sampler import CrossModalityIdentitySampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import IdentityCrossModalitySampler
from data.sampler import NormTripletSampler

from data.transform import TMS
from data.transform import NoTransform


def collate_fn(batch):
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size,
                     random_flip=False, random_crop=False, random_erase=False, color_jitter=False, padding=0,
                     tms=False, num_workers=4):
    # Data Transform - RGB ---------------------------------------------------------------------------------------------
    t = [T.Resize(image_size)]
    

    t.append(T.RandomChoice([
            T.RandomApply([T.ColorJitter(hue=0.20)], p=0.2) if color_jitter else NoTransform(),
            TMS(p=0.6) if tms else NoTransform(), # 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0 best is 0.6 
        ]))

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if random_crop:
        t.append(T.RandomCrop(image_size, padding=padding, fill=127))

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing(value=0, scale=(0.02, 0.30)))

    transform_rgb = T.Compose(t)

    # Data Transform - IR ----------------------------------------------------------------------------------------------
    t = [T.Resize(image_size)]

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if random_crop:
        t.append(T.RandomCrop(image_size, padding=padding, fill=127))

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing(value=0, scale=(0.02, 0.30)))

    transform_ir = T.Compose(t)

    # Dataset ----------------------------------------------------------------------------------------------------------
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform_rgb=transform_rgb, transform_ir=transform_ir)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform_rgb=transform_rgb, transform_ir=transform_ir)
    elif dataset == 'llcm':
        train_dataset = LLCMDataset(root, mode='train', transform_rgb=transform_rgb, transform_ir=transform_ir)
    else:
        raise NotImplementedError(f'Dataset - {dataset} is not supported')

    # DataSampler ------------------------------------------------------------------------------------------------------
    assert sample_method in ['none', 'norm_triplet',
                             'cross_modality_random',
                             'cross_modality_identity',
                             'identity_cross_modality']
    shuffle = False
    if sample_method == 'none':
        sampler = None
        shuffle = True
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'cross_modality_random':
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)
    elif sample_method == 'cross_modality_identity':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_cross_modality':
        batch_size = p_size * k_size
        sampler = IdentityCrossModalitySampler(train_dataset, p_size * k_size, k_size)

    # DataLoader -------------------------------------------------------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler,
                              shuffle=shuffle, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    return train_loader


def get_test_loader(dataset, root, batch_size, image_size, num_workers=4):
    # Data Transform - RGB ---------------------------------------------------------------------------------------------
    transform_rgb = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data Transform - IR ----------------------------------------------------------------------------------------------
    transform_ir = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset ----------------------------------------------------------------------------------------------------------
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform_rgb=transform_rgb, transform_ir=transform_ir)
        query_dataset = SYSUDataset(root, mode='query', transform_rgb=transform_rgb, transform_ir=transform_ir)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform_rgb=transform_rgb, transform_ir=transform_ir)
        query_dataset = RegDBDataset(root, mode='query', transform_rgb=transform_rgb, transform_ir=transform_ir)
    elif dataset == 'llcm':
        gallery_dataset = LLCMDataset(root, mode='gallery', transform_rgb=transform_rgb, transform_ir=transform_ir)
        query_dataset = LLCMDataset(root, mode='query', transform_rgb=transform_rgb, transform_ir=transform_ir)
    else:
        raise NotImplementedError(f'Dataset - {dataset} is not supported')

    # DataLoader -------------------------------------------------------------------------------------------------------
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader
