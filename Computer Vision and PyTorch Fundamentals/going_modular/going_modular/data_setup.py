"""
  Contains functionality for PyTorch DataLoader's for
  image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms,
    batch_size: int,
    num_workers: int = num_workers
):
  """Creates traning and testing DataLoaders.

  Takes in a train directory and test directory path and converts them into the
  PyTorch's Datasets and then into PyTorch's DataLoaders.

  Args:
    train_dir: Path to train directory.
    test_dir: Path to test directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: Number of workers for each DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=path/to/train_dir,
        test_dir=path/to/test_dir,
        transform=some_transform,
        batch_size=32,
        num_workers=4)
  """

  transform = transforms.Compose([
      transforms.Resize((64,64)),
      transforms.ToTensor()
  ])

  train_data = datasets.ImageFolder(train_dir, transform)
  test_data = datasets.ImageFolder(test_dir, transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)

  test_dataloader = DataLoader(test_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True)


  return train_dataloader, test_dataloader, class_names
