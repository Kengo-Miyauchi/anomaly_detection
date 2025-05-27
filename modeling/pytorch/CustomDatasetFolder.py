from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset


class DatasetFolderWithIgnoreFolder(DatasetFolder):
    def __init__(self, root, loader, ignore_folder_names, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super().__init__(root, loader, extensions, transform, target_transform, is_valid_file)
        classes, class_to_idx = self.find_classes(self.root)
        for ignore_name in ignore_folder_names:
            class_to_idx.pop(ignore_name)
        print(len(class_to_idx), class_to_idx)
        if is_valid_file is not None:
            extensions = None
        self.samples = self.make_dataset(self.root, class_to_idx, extensions)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class DatasetListFile(Dataset):
    def __init__(self, paths, loader, transform=None):
        super().__init__()
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = self.loader(self.paths[index])
        if len(data) == 0:
            print(index, self.paths[index])
        if self.transform:
            data = self.transform(data)
        return data


class DatasetListFileWithHeader(Dataset):
    def __init__(self, path_and_headers, loader, transform=None):
        super().__init__()
        self.paths, self.headers = path_and_headers
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = self.loader(self.paths[index])
        if len(data) == 0:
            print(index, self.paths[index])
        if self.transform:
            data = self.transform(data)
        return data, self.headers[index]
