import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd
import gzip


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        flip_img = np.random.rand() < self.p
        if flip_img:
            img = np.flip(img, axis=1)
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        H, W = img.shape[:2]
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding+1, size=2)
        shift_x += self.padding
        shift_y += self.padding
        if len(img.shape) == 3:
            img = np.pad(img, ((self.padding, self.padding), (self.padding,
                        self.padding), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            img = np.pad(img, ((self.padding, self.padding), (self.padding,
                        self.padding)), 'constant', constant_values=(0, 0))
        return img[shift_x:shift_x+H, shift_y:shift_y+W]

        


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        r = np.arange(len(dataset))
        if not self.shuffle:
            self.ordering = np.array_split(r, range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            r = np.random.permutation(np.arange(len(self.dataset)))
            self.ordering = np.array_split(r, range(self.batch_size, len(self.dataset), self.batch_size))
        self.ordering = iter(self.ordering)
        return self

    def __next__(self):
        mini_batch = next(self.ordering)
        return tuple(Tensor(x) for x in self.dataset[mini_batch])


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        with gzip.open(image_filename) as f:
            images = f.read()
            X = np.frombuffer(images, dtype=np.uint8,
                              offset=16).astype(np.float32)
            X /= 255.
            X = X.reshape((-1, 28, 28, 1))
        with gzip.open(label_filename) as f:
            labels = f.read()
            y = np.frombuffer(labels, dtype=np.uint8, offset=8)
        self.X = X
        self.y = y
        self.len = len(self.X)

    def __getitem__(self, index) -> object:
        if isinstance(index, int):
            x, y = self.apply_transforms(self.X[index]), self.y[index]
        else:
            x, y = [], []
            for idx in index:
                x.append(self.apply_transforms(self.X[idx]))
                y.append(self.y[idx])
        return x, y

    def __len__(self) -> int:
        return self.len


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        self.p = p
        self.transforms = transforms
        X = []
        y = []
        if train:
            for batch in range(1,6):
                with open(os.path.join(base_folder, f"data_batch_{batch}"), 'rb') as f:
                    data = pickle.load(f, encoding="bytes")
                    X.append(data[b"data"].astype(np.float32))
                    y.append(data[b"labels"])
        else:
            with open(os.path.join(base_folder, "test_batch"), 'rb') as f:
                    data = pickle.load(f, encoding="bytes")
                    X.append(data[b"data"].astype(np.float32))
                    y.append(data[b"labels"])
        self.X = np.concatenate(X, axis=0) / 255.
        self.y = np.concatenate(y, axis=0)
        self.len = len(self.X)
                
    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        if isinstance(index, int):
            X = self.X[index].reshape((3,32,32))
            y = self.y[index]
            X = self.apply_transforms(np.transpose(X, (1,2,0)))
            return np.transpose(X, (2,0,1)), y
        else:
            x_ret, y_ret = [], []
            for idx in index:
                X = self.X[idx].reshape((3,32,32))
                y = self.y[idx]
                X = self.apply_transforms(np.transpose(X, (1,2,0)))
                x_ret.append(np.transpose(X, (2,0,1)))
                y_ret.append(y)
            return x_ret, y_ret

    def __len__(self) -> int:
        return self.len


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION