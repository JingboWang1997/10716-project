from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def get_data():
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    return mnist_train, mnist_test


def get_numpy_data():
    trainset, testset = get_data()
    train_data_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    test_data_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
    train_data_tensor = next(iter(train_data_loader))
    test_data_tensor = next(iter(test_data_loader))
    return train_data_tensor[0].numpy(), train_data_tensor[1].numpy(), test_data_tensor[0].numpy(), test_data_tensor[1].numpy()