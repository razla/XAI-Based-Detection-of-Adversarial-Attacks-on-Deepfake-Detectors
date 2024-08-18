import time

from torch import autograd

from utils import *
import torch
import torch.nn as nn
import robust_transforms as rt
from dataset.transform import xception_default_data_transforms, EfficientNetB4ST_default_data_transforms
import random



def iterative_fgsm(input_img, model, model_type, cuda=True, max_iter=100, alpha=1 / 255.0, eps=16 / 255.0,
                   desired_acc=0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    if model_type == 'EfficientNetB4ST':
        post_function = nn.Sigmoid()
    else:
        post_function = nn.Softmax(dim=1)
    while iter_no < max_iter:
        prediction, output, logits = predict_with_model(input_var, model, model_type, cuda=cuda,
                                                        post_function=post_function)
        if model_type == 'EfficientNetB4ST':
            # repeated = logits.repeat(1, 2)
            # repeated[0][0] *= -1
            logits = nn.Softmax(dim=1)(logits)
        if (output[0][0] - output[0][1]) > desired_acc:
            break

        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(logits, target_var)
        if input_var.grad is not None:
            input_var.grad.data.zero_()  # just to ensure nothing funny happens
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)

        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)

        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print("L infinity norm", l_inf_norm, l_inf_norm * 255.0)

    meta_data = {
        'attack_iterations': iter_no,
        'l_inf_norm': l_inf_norm,
        'l_inf_norm_255': round(l_inf_norm * 255.0)
    }

    return input_var, meta_data


def black_box_attack(input_img, model, model_type,
                     cuda=True, max_iter=100, alpha=1 / 255.0,
                     eps=16 / 255.0, desired_acc=0.90,
                     transform_set={"gauss_blur", "translation"}):
    def _get_transforms(apply_transforms={"gauss_noise", "gauss_blur", "translation", "resize"}):

        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: rt.add_gaussian_noise(x, 0.01, cuda=cuda),
            ]

        if "gauss_blur" in apply_transforms:
            kernel_size = random.randint(3, 6)
            if kernel_size % 2 == 0:
                kernel_size = kernel_size - 1
            sigma = random.randint(5, 7)
            transform_list += [
                lambda x: rt.gaussian_blur(x, kernel_size=(kernel_size, kernel_size), sigma=(sigma * 1., sigma * 1.),
                                           cuda=cuda)
            ]

        if "translation" in apply_transforms:
            x_translate = random.randint(-20, 20)
            y_translate = random.randint(-20, 20)

            transform_list += [
                lambda x: rt.translate_image(x, x_translate, y_translate, cuda=cuda),
            ]

        if "resize" in apply_transforms:
            compression_factor = random.randint(4, 6) / 10.0
            transform_list += [
                lambda x: rt.compress_decompress(x, compression_factor, cuda=cuda),
            ]

        return transform_list

    def _find_nes_gradient(input_var, transform_functions, model, model_type, num_samples=20, sigma=0.001):
        g = 0
        _num_queries = 0
        for sample_no in range(num_samples):
            for transform_func in transform_functions:
                rand_noise = torch.randn_like(input_var)
                img1 = input_var + sigma * rand_noise
                img2 = input_var - sigma * rand_noise

                prediction1, probs_1, _ = predict_with_model(transform_func(img1), model, model_type, cuda=cuda)

                prediction2, probs_2, _ = predict_with_model(transform_func(img2), model, model_type, cuda=cuda)

                _num_queries += 2
                g = g + probs_1[0][0] * rand_noise
                g = g - probs_2[0][0] * rand_noise
                g = g.data.detach()

                del rand_noise
                del img1
                del prediction1, probs_1
                del prediction2, probs_2

        return (1. / (2. * num_samples * len(transform_functions) * sigma)) * g, _num_queries

    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0

    # give it a warm start by crafting by fooling without any transformations -> easier
    warm_start_done = False
    num_queries = 0
    while iter_no < max_iter:

        if not warm_start_done:
            _, output, _ = predict_with_model(input_var, model, model_type, cuda=cuda)
            num_queries += 1
            if output[0][0] > desired_acc:
                warm_start_done = True

        if warm_start_done:
            # choose all transform functions
            transform_functions = _get_transforms(transform_set)
        else:
            transform_functions = _get_transforms({})  # returns identity function

        all_fooled = True
        print("Testing transformation outputs", iter_no)
        for transform_fn in transform_functions:
            _, output, _ = predict_with_model(transform_fn(input_var), model, model_type, cuda=cuda)
            num_queries += 1
            print(output)
            if output[0][0] < desired_acc:
                all_fooled = False

        print("All transforms fooled:", all_fooled, "Warm start done:", warm_start_done)
        if warm_start_done and all_fooled:
            break

        step_gradient_estimate, _num_grad_calc_queries = _find_nes_gradient(input_var, transform_functions, model,
                                                                            model_type)
        num_queries += _num_grad_calc_queries
        step_adv = input_var.detach() + alpha * torch.sign(step_gradient_estimate.data.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)

        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)

        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print("L infinity norm", l_inf_norm, l_inf_norm * 255.0)

    meta_data = {
        'num_network_queries': num_queries,
        'attack_iterations': iter_no,
        'l_inf_norm': l_inf_norm,
        'l_inf_norm_255': round(l_inf_norm * 255.0)
    }

    return input_var, meta_data


