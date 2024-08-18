import torch
import cv2
import numpy as np
from PIL import Image as pil_image
import torch.nn as nn
from torchvision import transforms

from dataset.transform import *


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, model_type, cuda=True, legacy=False):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    if not legacy:
        # only conver to tensor here,
        # other transforms -> resize, normalize differentiable done in predict_from_model func
        # same for meso, xception
        if model_type == 'meso' or model_type == 'xception':
            preprocess = xception_default_data_transforms['to_tensor']
            preprocessed_image = preprocess(pil_image.fromarray(image))
        elif model_type == 'EfficientNetB4ST':
            # preprocess = EfficientNetB4ST_default_data_transforms['to_tensor']
            # normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # preprocess = get_transformer('scale', 224, normalizer, train=False)
            # preprocessed_image = preprocess(image=image)['image']
            preprocess = EfficientNetB4ST_default_data_transforms['to_tensor']
            preprocessed_image = preprocess(pil_image.fromarray(image))

    else:
        if model_type == "xception":
            preprocess = xception_default_data_transforms['test']
            preprocessed_image = preprocess(pil_image.fromarray(image))


        elif model_type == 'EfficientNetB4ST':
            # normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # preprocess = get_transformer('scale', 224, normalizer, train=False)
            # preprocessed_image = preprocess(image=image)['image']
            preprocess = EfficientNetB4ST_default_data_transforms['test']
            preprocessed_image = preprocess(pil_image.fromarray(image))

    preprocessed_image = preprocessed_image.unsqueeze(0)

    if cuda:
        preprocessed_image = preprocessed_image.cuda()

    preprocessed_image.requires_grad = True
    return preprocessed_image


def un_preprocess_image(image, size):
    """
    Tensor to PIL image and RGB to BGR
    """

    image.detach()
    new_image = image.squeeze(0)
    new_image = new_image.detach().cpu()

    undo_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size)
    ])

    new_image = undo_transform(new_image)
    new_image = np.array(new_image)

    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    return new_image


def check_attacked(preprocessed_image, xai_map, model, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of an input image. Performs resizing and normalization before feeding in image.

    :param image: torch tenosr (bs, c, h, w)
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = attacked, 0 = real), output probs, logits
    """

    # Model prediction

    # differentiable resizing: doing resizing here instead of preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    normalized_image = transform(preprocessed_image)
    normalized_xai = transform(xai_map)
    logits = model(normalized_image.float().cuda(), normalized_xai.float().cuda())
    output = post_function(logits)
    _, prediction = torch.max(output, 1)  # argmax
    prediction = float(prediction.cpu().numpy())
    output = output.detach().cpu().numpy().tolist()
    # print ("prediction", prediction)
    # print ("output", output)
    return int(prediction), output, logits


def predict_with_model_legacy(image, model, model_type, post_function=nn.Softmax(dim=1),
                              cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, model_type, cuda, legacy=True)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)
    if model_type == 'EfficientNetB4ST':
        fake_pred = output[0][1].item()
        real_pred = 1 - fake_pred
        output = np.array([real_pred, fake_pred])
        prediction = float(np.argmax(output))
        output = [output.tolist()]
    # Cast to desired
    else:
        _, prediction = torch.max(output, 1)  # argmax
        prediction = float(prediction.cpu().numpy())
        output = output.detach().cpu().numpy().tolist()
    return int(prediction), output


def calculate_xai_map(cropped_face, model, model_type, xai_calculator, xai_method, cuda=True):
    preprocessed_image = preprocess_image(cropped_face, model_type)
    prediction, output = predict_with_model_legacy(cropped_face, model, model_type, post_function=nn.Softmax(dim=1),
                                                   cuda=cuda)
    if xai_method == 'IntegratedGradients':
        xai_img = xai_calculator.attribute(preprocessed_image, target=prediction, internal_batch_size=1)
    else:
        xai_img = xai_calculator.attribute(preprocessed_image, target=prediction)
    return xai_img


def predict_with_model(preprocessed_image, model, model_type, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of an input image. Performs resizing and normalization before feeding in image.

    :param image: torch tenosr (bs, c, h, w)
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real), output probs, logits
    """

    # Model prediction

    # differentiable resizing: doing resizing here instead of preprocessing
    if model_type == "xception":
        resized_image = nn.functional.interpolate(preprocessed_image, size=(299, 299), mode="bilinear",
                                                  align_corners=True)
        norm_transform = xception_default_data_transforms['normalize']
        normalized_image = norm_transform(resized_image)
    elif model_type == 'EfficientNetB4ST':
        resized_image = nn.functional.interpolate(preprocessed_image, size=(224, 224), mode="bilinear",
                                                  align_corners=True)
        norm_transform = EfficientNetB4ST_default_data_transforms['normalize']
        normalized_image = norm_transform(resized_image)
        # normalized_image = preprocessed_image

    logits = model(normalized_image)
    output = post_function(logits)

    if model_type == 'EfficientNetB4ST':
        fake_pred = output[0][1].item()
        real_pred = 1 - fake_pred
        output = np.array([real_pred, fake_pred])
        prediction = float(np.argmax(output))
        output = [output.tolist()]
    else:
        # Cast to desired
        _, prediction = torch.max(output, 1)  # argmax
        prediction = float(prediction.cpu().numpy())
        output = output.detach().cpu().numpy().tolist()
    # print ("prediction", prediction)
    # print ("output", output)
    return int(prediction), output, logits
