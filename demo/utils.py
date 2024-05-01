import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import requests
import cv2

from torch.nn.functional import cosine_similarity, pairwise_distance

import models_mae


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, mask_ratio=0.75):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()

    return x, im_masked, y, im_paste


def get_features_from_image(img, model, mask_ratio=0):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    x = x.float()

    # run MAE
    # loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    # y = model.unpatchify(y)
    # y = torch.einsum('nchw->nhwc', y).detach().cpu()

    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder_no_mask(x, mask_ratio=mask_ratio)
        embed = model.patch_embed(x)

    return embed, latent, mask, ids_restore


def load_image_from_url(url):
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std
    return img

def load_image_from_file(file):
    img = Image.open(file)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std
    return img

def place_image(img, img2, size=(32, 32), position='topleft'):
    img3 = img.copy()

    if isinstance(size, int):
        size = (size, size)

    # Resize img2 to specified size
    img2_resized = cv2.resize(img2, size)

    # Calculate the position where to place img2_resized
    if position == 'topleft':
        img3[0:size[1], 0:size[0]] = img2_resized
    elif position == 'topright':
        img3[0:size[1], img3.shape[1]-size[0]:img3.shape[1]] = img2_resized
    elif position == 'bottomleft':
        img3[img3.shape[0]-size[1]:img3.shape[0], 0:size[0]] = img2_resized
    elif position == 'bottomright':
        img3[img3.shape[0]-size[1]:img3.shape[0], img3.shape[1]-size[0]:img3.shape[1]] = img2_resized

    return img3


def get_features_from_image(img, model, mask_ratio=0):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    x = x.float()

    # run MAE
    # loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    # y = model.unpatchify(y)
    # y = torch.einsum('nchw->nhwc', y).detach().cpu()

    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder_no_mask(x, mask_ratio=mask_ratio)
        # embed = model.patch_embed(x)

    # return embed, latent, mask, ids_restore

    # remove the classification token
    return latent, mask, ids_restore

def reconstruct_image_from_features(latent, ids_restore, model, show=False):
    # add the classification token
    # latent = torch.cat([torch.zeros(latent.shape[0], 1, latent.shape[2]), latent], dim=1)

    with torch.no_grad():
        y = model.forward_decoder(latent, ids_restore)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

    if show:
        show_image(y[0])
    return y

def calculate_similarity(x1, x2, img1=None, img2=None, type='cosine', show=False):
    # Reshape the tensors
    x1 = x1.view(14, 14, 768)
    x2 = x2.view(14, 14, 768)

    plot_vmin, plot_vmax = None, None

    # Calculate the similarity
    if type == 'cosine':
        similarity = cosine_similarity(x1, x2, dim=2)
        plot_vmin, plot_vmax = 0.0, 1.0

    elif type == 'l1':
        similarity = 1 / (1 + pairwise_distance(x1, x2, p=1, keepdim=False))
    elif type == 'l2':
        similarity = 1 / (1 + pairwise_distance(x1, x2, p=2, keepdim=False))

    # if show:
    #     plt.imshow(similarity.numpy(), cmap='hot', interpolation='nearest', vmin=plot_vmin, vmax=plot_vmax )
    #     plt.colorbar()
    #     plt.show()

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        if img1 is not None:
            axs[0].imshow(torch.clip((torch.tensor(img1) * imagenet_std + imagenet_mean) * 255, 0, 255).int())
            axs[0].set_title('Image 1')

        if img2 is not None:
            axs[1].imshow(torch.clip((torch.tensor(img2) * imagenet_std + imagenet_mean) * 255, 0, 255).int())
            axs[1].set_title('Image 2')

        im = axs[2].imshow(similarity, cmap='hot', interpolation='nearest', vmin=plot_vmin, vmax=plot_vmax)
        axs[2].set_title('Similarity')
        # fig.colorbar(im, ax=axs[2])

        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

        plt.show()

    return similarity


def calculate_similarity_images(img1, img2, model, type='cosine', show=False):
    # Get the features
    x1, _, ids_restore = get_features_from_image(img1, model)
    x2, _, _ = get_features_from_image(img2, model)

    # Calculate the similarity
    # use x1[0, 1:] to remove the classification token
    similarity = calculate_similarity(x1[0, 1:], x2[0, 1:], img1=img1, img2=img2, type=type, show=show)

    return similarity, x1, x2, ids_restore

def combine_image_in_latent_space(img1, img2, model, alpha=0.5, show=False):
    # Get the features
    x1, _, ids_restore = get_features_from_image(img1, model)
    x2, _, _ = get_features_from_image(img2, model)

    # Combine the features
    latent = alpha * x1 + (1 - alpha) * x2

    # Reconstruct the image
    y = reconstruct_image_from_features(latent, ids_restore, model, show=False)

    # show the three images in subplots
    if show:
        plt.rcParams['figure.figsize'] = [10, 3]
        plt.subplot(1, 3, 1)
        show_image(torch.tensor(img1), "Image 1")

        plt.subplot(1, 3, 2)
        show_image(torch.tensor(img2), "Image 2")

        plt.subplot(1, 3, 3)
        show_image(y[0], "Combined Image")

        plt.show()

    return y, latent


# temp = x1[0,1:, 100].reshape(14, 14)
# # repeat the tensor to 3 channels
# temp = torch.stack([temp, temp, temp], dim=2)
# temp = temp.detach().cpu().numpy()
# plt.rcParams['figure.figsize'] = [3, 3]
# plt.imshow(temp)