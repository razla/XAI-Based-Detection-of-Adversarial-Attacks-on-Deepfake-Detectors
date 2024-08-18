import argparse
import os

from torchmetrics import ROC
from tqdm import tqdm
from train import CustomResNet50, calculate_accuracy
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.transform import ImageXaiFolder

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--attacked_detector_model_path', '-mi', type=str, default=None)
    p.add_argument('--deepfake_detector_model_type', '-mt', type=str, default="xception")
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--real_dataset_path', '-rd', type=str, default=None)
    p.add_argument('--attacked_dataset_path', '-ad', type=str, default=None)
    p.add_argument('--xai_method', '-xm', type=str, default='InputXGradient')
    p.add_argument('--batch_size', '-b', type=int, default=16)
    args = p.parse_args()
    output_path = args.output_path
    model_path = args.attacked_detector_model_path
    model_type = args.deepfake_detector_model_type
    real_dataset_path = args.real_dataset_path
    attacked_dataset_path = args.attacked_dataset_path
    xai_method = args.xai_method
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Testing attacked model trained on {xai_method} dataset defending {model_type} ')

    weights = ResNet50_Weights.DEFAULT
    model = CustomResNet50(weights=weights)
    model.to(device)
    output_dir = os.path.join(output_path, model_type, xai_method)
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, f'model_acc.txt'), 'w')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    activation_function = nn.Softmax(dim=1)

    test_dataset = ImageXaiFolder(original_path=os.path.join(real_dataset_path, model_type, 'Frames'),
                                  original_xai_path=os.path.join(real_dataset_path, model_type, xai_method),
                                  attacked_path=os.path.join(attacked_dataset_path, model_type, 'Frames'),
                                  attacked_xai_path=os.path.join(attacked_dataset_path, model_type, xai_method),
                                  transform=transform,
                                  black_xai=False,
                                  black_img=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_accuracy = 0.0
    batch_bar = tqdm(total=len(test_loader))
    concatenated_preds = torch.empty(0).to(device)
    concatenated_labels = torch.empty(0).to(device)
    r = ROC(task="multiclass", num_classes=2)
    with torch.no_grad():
        for test_images, test_xais, test_labels in test_loader:
            test_images = test_images.to(device)
            test_xais = test_xais.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images.float(), test_xais.float())
            concatenated_preds = torch.cat((concatenated_preds, activation_function(test_outputs)))
            concatenated_labels = torch.cat((concatenated_labels, test_labels))
            # test_outputs = model(test_xais.float())
            # test_outputs = activation_function(test_outputs)
            accuracy = calculate_accuracy(activation_function(test_outputs), test_labels)
            total_accuracy += accuracy
            batch_bar.update(1)
    fpr, tpr, threshold = r(concatenated_preds,
                            torch.argmax(concatenated_labels.squeeze().to(torch.long), dim=1))
    save_roc(fpr, tpr, output_dir)
    batch_bar.close()
    avg_accuracy = total_accuracy / len(test_loader)
    print(f'Accuracy: {avg_accuracy}')
    f.write('Accuracy: ' + str(avg_accuracy))
    f.close()

    pass


def save_roc(fpr, tpr, working_dir):
    import matplotlib.pyplot as plt
    fpr = fpr[1].cpu().detach().numpy()
    tpr = tpr[1].cpu().detach().numpy()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.show()
    plt.savefig(f'{working_dir}/ROC_graph.png')
    pass


if __name__ == '__main__':
    main()
