import sys
import argparse
from os.path import join
import dlib

from tqdm import tqdm

from network.models import model_selection

from attack_algos import iterative_fgsm, black_box_attack, \
    predict_with_model as predict_with_model_attack_algos
from utils import *

import json
from captum.attr import IntegratedGradients, InputXGradient, GuidedBackprop, Saliency


def create_adversarial_video(video_path, deepfake_detector_model_path, deepfake_detector_model_type, output_path,
                             xai_method=None, attacked_detector_model_path=None,
                             start_frame=0, end_frame=None, attack="iterative_fgsm", eps=16 / 255,
                             compress=True, cuda=True, showlabel=True):
    """
    Reads a video and evaluates a subset of frames with the detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param compress:
    :param showlabel:
    :param adaptive_attack:
    :param xai_method:
    :param atttacked_detector_model_path:
    :param video_path: path to video file
    :param deepfake_detector_model_path: path to deepfake_detector_model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    xai_map = None
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_path = video_path.replace('\\', '/') if '\\' in video_path else video_path
    video_fn = video_path.split('/')[-1].split('.')[0] + '.avi'
    os.makedirs(output_path, exist_ok=True)

    if compress:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')  # Chnaged to HFYU because it is lossless

    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load deepfake detector deepfake_detector_model
    if deepfake_detector_model_path is not None:
        if not cuda:
            deepfake_detector_model = torch.load(deepfake_detector_model_path, map_location="cpu")
        else:
            if deepfake_detector_model_type == 'xception':
                deepfake_detector_model = torch.load(deepfake_detector_model_path)
            elif deepfake_detector_model_type == 'EfficientNetB4ST':
                deepfake_detector_model = model_selection('EfficientNetB4ST', 2)
                weights = torch.load(deepfake_detector_model_path)
                deepfake_detector_model.load_state_dict(weights)
            else:
                raise f"{deepfake_detector_model_type} not supported"
        print('Model found in {}'.format(deepfake_detector_model_path))
    else:
        print('No deepfake_detector_model found, initializing random deepfake_detector_model.')
    if cuda:
        print("Converting mode to cuda")
        deepfake_detector_model = deepfake_detector_model.eval().cuda()
        for param in deepfake_detector_model.parameters():
            param.requires_grad = True
        print("Converted to cuda")

    # raise Exception()
    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame - start_frame)

    if attack.find('adaptive') != -1:
        metrics = {
            'total_fake_real_frames': 0,
            'total_real_real_frames': 0,
            'total_fake_attacked_frames': 0,
            'total_real_attacked_frames': 0,
            'total_frames': 0,
            'precent_fake_real': 0,
            'percent_fake_attacked': 0,
            'percent_real_real': 0,
            'percent_real_attacked': 0,
            'probs_list': [],
            'attacked_detector_probs_list': [],
            'attack_meta_data': [],
        }
    else:
        metrics = {
            'total_fake_frames': 0,
            'total_real_frames': 0,
            'total_frames': 0,
            'percent_fake_frames': 0,
            'probs_list': [],
            'attack_meta_data': [],
        }

    if deepfake_detector_model_type == 'EfficientNetB4ST':
        post_function = nn.Sigmoid()
    else:
        post_function = nn.Softmax(dim=1)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            processed_image = preprocess_image(cropped_face, deepfake_detector_model_type, cuda=cuda)

            # Attack happening here

            # white-box attacks
            if attack == "fgsm":
                perturbed_image, attack_meta_data = iterative_fgsm(processed_image,
                                                                   deepfake_detector_model,
                                                                   deepfake_detector_model_type,
                                                                   max_iter=1,
                                                                   eps=eps,
                                                                   cuda=cuda)
            elif attack == "pgd":
                perturbed_image, attack_meta_data = iterative_fgsm(processed_image,
                                                                   deepfake_detector_model,
                                                                   deepfake_detector_model_type,
                                                                   max_iter=100,
                                                                   eps=eps,
                                                                   cuda=cuda)

            # black-box attacks
            elif attack == "nes":
                perturbed_image, attack_meta_data = black_box_attack(processed_image,
                                                                     deepfake_detector_model,
                                                                     deepfake_detector_model_type,
                                                                     eps=eps, cuda=cuda, transform_set={},
                                                                     desired_acc=0.999)

            # Undo the processing
            unpreprocessed_image = un_preprocess_image(perturbed_image, size)
            image[y:y + size, x:x + size] = unpreprocessed_image
            cropped_face = image[y:y + size, x:x + size]
            processed_image = preprocess_image(cropped_face, deepfake_detector_model_type, cuda=cuda)
            prediction, output, logits = predict_with_model_attack_algos(processed_image, deepfake_detector_model,
                                                                         deepfake_detector_model_type, cuda=cuda,
                                                                         post_function=post_function)
            print(">>>>Prediction for frame no. {}: {}".format(frame_num, output))

            prediction, output = predict_with_model_legacy(cropped_face, deepfake_detector_model,
                                                           deepfake_detector_model_type, cuda=cuda,
                                                           post_function=post_function)
            print(">>>>Prediction LEGACY for frame no. {}: {}".format(frame_num, output))

            label = 'fake' if prediction == 1 else 'real'
            if label == 'fake':
                metrics['total_fake_frames'] += 1.
            else:
                metrics['total_real_frames'] += 1.

            metrics['total_frames'] += 1.
            metrics['probs_list'].append(output[0])
            metrics['attack_meta_data'].append(attack_meta_data)

        if showlabel:
            # Text and bb
            # print a bounding box in the generated video
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]

            cv2.putText(image, str(output_list) + '=>' + label, (x, y + h + 30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        writer.write(image)
    pbar.close()

    if attack.find('adaptive') != -1:
        metrics['percent_fake_real'] = metrics['total_fake_real_frames'] / metrics['total_frames']
        metrics['percent_fake_attacked'] = metrics['total_fake_attacked_frames'] / metrics['total_frames']
        metrics['percent_real_real'] = metrics['total_real_real_frames'] / metrics['total_frames']
        metrics['percent_real_attacked'] = metrics['total_real_attacked_frames'] / metrics['total_frames']
    else:
        metrics['percent_fake_frames'] = metrics['total_fake_frames'] / metrics['total_frames']

    with open(join(output_path, video_fn.replace(".avi", "_metrics_attack.json")), "w") as f:
        f.write(json.dumps(metrics))
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--deepfake_detector_model_path', '-mi', type=str, default=None)
    p.add_argument('--deepfake_detector_model_type', '-mt', type=str, default="xception")
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--attack', '-a', type=str, default="pgd")
    p.add_argument('--eps', type=float, default=16 / 255)
    p.add_argument('--compress', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--showlabel', action='store_true')  # add face labels in the generated video

    args = p.parse_args()
    args.output_path += f'/{args.attack}/{args.deepfake_detector_model_type}'
    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        create_adversarial_video(**vars(args))
    else:
        videos = os.listdir(video_path)
        videos = [video for video in videos if (video.endswith(".mp4") or video.endswith(".avi"))]
        pbar_global = tqdm(total=len(videos))
        for video in videos:
            if os.path.exists(os.path.join(args.output_path, os.path.splitext(video)[0] + '_metrics_attack.json')):
                print(f'Adversarial video already exists for {video}')
                continue
            args.video_path = join(video_path, video)
            # blockPrint()
            create_adversarial_video(**vars(args))
            # enablePrint()
            pbar_global.update(1)
        pbar_global.close()
