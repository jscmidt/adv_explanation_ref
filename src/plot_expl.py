import argparse
import torch
import torchvision
import numpy as np

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, load_image, make_dir


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--img', type=str, default='../data/collie4.jpeg', help='image net file to run attack on')
    argparser.add_argument('--x', type=str, default='', help="tensor to calculate expl from")
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--output_dir', type=str, default='../output/', help='directory to save results to')
    argparser.add_argument('--betas', nargs='+', help='beta values for softplus explanations', type=float,
                           default=[10, 3, 1])
    argparser.add_argument('--method', help='algorithm for expls',
                           choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                    'pattern_attribution', 'grad_times_input', 'gradcam'],
                           default='gradient')
    argparser.add_argument('--model_name', type=str, default='vgg16', choices=['vgg16', 'maxvit_t'])
    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    method = getattr(ExplainingMethod, args.method)

    # load model
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])
    if args.model_name == 'vgg16':
        model_arch = torchvision.models.vgg16(pretrained=True)
    elif args.model_name == 'maxvit_t':
        model_arch = torchvision.models.maxvit_t(pretrained=True)
        
    model = ExplainableNet(model_arch, data_mean=data_mean, data_std=data_std, beta=None)
    if method == ExplainingMethod.pattern_attribution:
        if args.model_name != 'vgg16':
            print('Pattern attribution is only available for VGG16')
            exit()
        model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    model = model.eval().to(device)

    # load images
    x = load_image(data_mean, data_std, device, args.img)
    if len(args.x) > 0:
        x = torch.load(args.x).to(device)

    # produce expls
    expls = []
    expl, _, org_idx = get_expl(model, x, method)
    expls.append(expl)
    captions = ["Image", "Expl. with ReLU"]

    # Beta sweeps only apply to the custom VGG16 wrapper layers.
    # For MaxViT, `change_beta()` does not alter internal activations (GELU/SiLU),
    # so producing multiple beta plots would be misleading.
    if args.model_name == 'vgg16':
        for beta in args.betas:
            model.change_beta(beta)
            expl, _, _ = get_expl(model, x, method, desired_index=org_idx)
            expls.append(expl)
            captions.append(f'Expl. with softplus \nbeta={beta}')
    else:
        captions = ["Image", "Explanation"]

    # save results
    output_dir = make_dir(args.output_dir)
    plot_overview([x], expls, data_mean, data_std, captions=captions,
                  filename=f"{output_dir}expls_{args.method}_{args.model_name}.png", images_per_row=len(expls)+1)


if __name__ == "__main__":
    main()
