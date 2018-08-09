import os
import errno
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torchvision.datasets.mnist import read_image_file, read_label_file

MNIST_PATH = '/home/stensootla/projects/datasets/mnist'


class MnistSingleDigit(Dataset):
  urls = [
      'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
      'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
      'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
      'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
  ]
  raw_folder = 'raw'
  processed_folder = 'processed'
  training_file = 'training.pt'
  test_file = 'test.pt'
  
  def __init__(self, root, digit, train=True, transform=None, download=False):
    assert digit >= 0 and digit <= 9
    self.digit = digit

    self.root = os.path.expanduser(root)
    self.transform = transform
    self.train = train  # training set or test set

    if download:
      self.download()

    if not self._check_exists():
      raise RuntimeError('Dataset not found.' +
                         ' You can use download=True to download it')

    if self.train:
      all_train_data, all_train_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.training_file))
      indices = torch.nonzero(all_train_labels == digit).squeeze()
      self.train_data = torch.index_select(all_train_data, 0, indices)
    else:
      all_test_data, all_test_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.test_file))
      indices = torch.nonzero(all_test_labels == digit).squeeze()
      self.test_data = torch.index_select(all_test_data, 0, indices)

  def __getitem__(self, index):
    if self.train: img = self.train_data[index]
    else: img = self.test_data[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img.numpy(), mode='L')

    if self.transform is not None:
      img = self.transform(img)

    return img

  def __len__(self):
    if self.train:
      return len(self.train_data)
    else:
      return len(self.test_data)

  def _check_exists(self):
    return os.path.exists(os.path.join(self.root, 
      self.processed_folder, self.training_file)) and \
        os.path.exists(os.path.join(self.root, \
          self.processed_folder, self.test_file))

  def download(self):
    """Download the MNIST data if it doesn't exist in processed_folder already."""
    from six.moves import urllib
    import gzip

    if self._check_exists():
      return

    # download files
    try:
      os.makedirs(os.path.join(self.root, self.raw_folder))
      os.makedirs(os.path.join(self.root, self.processed_folder))
    except OSError as e:
      if e.errno == errno.EEXIST:
        pass
      else:
        raise

    for url in self.urls:
      filename = url.rpartition('/')[2]
      file_path = os.path.join(self.root, self.raw_folder, filename)
      download_url(url, root=os.path.join(self.root, self.raw_folder),
                   filename=filename, md5=None)
      with open(file_path.replace('.gz', ''), 'wb') as \
          out_f, gzip.GzipFile(file_path) as zip_f:
        out_f.write(zip_f.read())
      os.unlink(file_path)

    # process and save as torch files
    print('Processing...')

    training_set = (
      read_image_file(os.path.join(self.root, self.raw_folder, \
        'train-images-idx3-ubyte')),
      read_label_file(os.path.join(self.root, self.raw_folder, \
        'train-labels-idx1-ubyte'))
    )
    test_set = (
      read_image_file(os.path.join(self.root, self.raw_folder, \
        't10k-images-idx3-ubyte')),
      read_label_file(os.path.join(self.root, self.raw_folder, \
        't10k-labels-idx1-ubyte'))
    )
    with open(os.path.join(self.root, self.processed_folder, \
        self.training_file), 'wb') as f:
      torch.save(training_set, f)
    with open(os.path.join(self.root, self.processed_folder, \
        self.test_file), 'wb') as f:
      torch.save(test_set, f)
    print('Done!')


def get_mnist_loader(digit, batch_size, train=False):
  all_transforms = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor()
  ])
  dataset = MnistSingleDigit(MNIST_PATH, digit=digit, train=train, 
    download=True, transform=all_transforms)
  mnist_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return mnist_loader

