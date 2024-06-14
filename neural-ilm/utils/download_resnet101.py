import torchvision.models as models


def run():
    resnet101 = models.resnet101(pretrained=True)


if __name__ == '__main__':
    run()
