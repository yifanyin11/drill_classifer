from UI import SimpleUI
from predictor import Predictor
from models import ResNet, VGG16


if __name__ == "__main__":
    ui = SimpleUI()
    path = ui.choose_directory()
    net = VGG16(num_classes=9)
    pred = Predictor(path, net)
    pred.inference()

    input("\n\nPress any key to terminate the program")