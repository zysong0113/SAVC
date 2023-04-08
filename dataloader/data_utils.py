import numpy as np
import torch
import torchvision.transforms as transforms
from dataloader.sampler import CategoriesSampler
from augmentations.constrained_cropping import CustomMultiCropDataset, CustomMultiCropping

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    args.Dataset=Dataset
    return args

def get_transform(args):
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
    if args.dataset == 'cub200':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    assert (len(args.size_crops) == 2)
    crop_transform = CustomMultiCropping(size_large=args.size_crops[0],
                                         scale_large=(args.min_scale_crops[0], args.max_scale_crops[0]),
                                         size_small=args.size_crops[1],
                                         scale_small=(args.min_scale_crops[1], args.max_scale_crops[1]),
                                         N_large=args.num_crops[0], N_small=args.num_crops[1],
                                         condition_small_crops_on_key=args.constrained_cropping)

    if len(args.auto_augment) == 0:
        print('No auto augment - Apply regular moco v2 as secondary transform')
        secondary_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),    
            transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize])

    else:
        from utils.auto_augment.auto_augment import AutoAugment
        from utils.auto_augment.random_choice import RandomChoice
        print('Auto augment - Apply custom auto-augment strategy')
        counter = 0
        secondary_transform = []

        for i in range(len(args.size_crops)):
            for j in range(args.num_crops[i]):
                if not counter in set(args.auto_augment):
                    print('Crop {} - Apply regular secondary transform'.format(counter))
                    secondary_transform.extend([transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
#                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.ToTensor(),
                        normalize])])

                else:
                    print('Crop {} - Apply auto-augment/regular secondary transform'.format(counter))
                    trans1 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        AutoAugment(),
                        transforms.ToTensor(),
                        normalize])

                    trans2 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
#                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        transforms.ToTensor(),
                        normalize])

                    secondary_transform.extend([RandomChoice([trans1, trans2])])

                counter += 1
    return crop_transform, secondary_transform

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    crop_transform, secondary_transform = get_transform(args)
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, index=class_index, base_sess=True,
                                         crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,index=class_index, base_sess=True,
                                       crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_sess=True,
                                             crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args,session):
    crop_transform, secondary_transform = get_transform(args)
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False, index=class_index, base_sess=False,                                              crop_transform=crop_transform, secondary_transform=secondary_transform)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path,
                                       crop_transform=crop_transform, secondary_transform=secondary_transform)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path,
                                             crop_transform=crop_transform, secondary_transform=secondary_transform)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list