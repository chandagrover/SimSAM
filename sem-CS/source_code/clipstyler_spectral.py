import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import StyleNet
import utils
import clip
import torch.nn.functional as F
from template import imagenet_templates
import os

# from IPython.display import display
# from argparse import Namespace


import argparse
from torchvision.transforms.functional import adjust_contrast
from torchvision import utils as vutils


def img_denormalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = image * std + mean
    return image

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image - mean) / std
    return image

def clip_normalize(image,device):
    image = F.interpolate(image,size=224, mode='bicubic', align_corners=False)
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image-mean)/std
    return image

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2

def compose_text_with_templates(text, templates=imagenet_templates):
    return [template.format(text) for template in templates]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# VGG = models.vgg19(pretrained=True).features
VGG = models.vgg19(pretrained=False).features
VGG.to(device)
for parameter in VGG.parameters():
    parameter.requires_grad_(False)

clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)


source = "Night"
crop_size = 128
training_iterations = 400
# text = "Starry Night by Vincent van gogh"
# text = "The great wave off Wanagawa by Hokusai"
# text = "The scream by edvard munch"


# image_dir = "test_set/images/lena.png"


# training_args = {
#     "lambda_tv": 2e-3,
#     "lambda_patch": 9000,
#     "lambda_dir": 500,
#     "lambda_c": 150,
#     "lambda_patch_context":1000,
#     "crop_size": 128,
#     "num_crops": 64,
#     "img_size": 512,
#     "max_step": training_iterations,
#     "lr": 5e-4,
#     "thresh": 0.7,
#     "content_path": image_dir,
#     "text": text
# }
# args = Namespace(**training_args)

parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, default="dog_resized",
                    help='Image resolution')

parser.add_argument('--content_path', type=str, default="test_set/images/dog_resized.png",
                    help='Image resolution')
parser.add_argument('--img_size', type=str, default=512,
                    help='Image size')
parser.add_argument('--segmentedImage_path', type=str, default="segmaps/deep_spectral/single_region_segmentation/crf_deep_spectral/dog_resized_segCRF.npy",
                    help='Segmented Image Path')
parser.add_argument('--content_name', type=str, default="face",
                    help='Content Name')
parser.add_argument('--exp_name', type=str, default="exp1",
                    help='Exepriment Name')
parser.add_argument('--object_category', type=int, default=255,
                    help='Object Category')
parser.add_argument('--text', type=str, default="Fire",
                    help='Text Description')
parser.add_argument('--lambda_tv', type=float, default=2e-3,
                    help='total variation loss parameter')
parser.add_argument('--lambda_patch', type=float, default=9000,
                    help='PatchCLIP loss parameter')
parser.add_argument('--lambda_dir', type=float, default=500,
                    help='directional loss parameter')
parser.add_argument('--lambda_c', type=float, default=150,
                    help='content loss parameter')
# parser.add_argument('--lambda_patch_context', type=float, default=1000,
#                     help='Patch Context parameter')
parser.add_argument('--crop_size', type=int, default=128,
                    help='cropped image size')
parser.add_argument('--num_crops', type=int, default=64,
                    help='number of patches')
parser.add_argument('--img_width', type=int, default=512,
                    help='size of images')
parser.add_argument('--img_height', type=int, default=512,
                    help='size of images')
parser.add_argument('--max_step', type=int, default=400,
                    help='Number of Iterations')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Number of domains')
parser.add_argument('--thresh', type=float, default=0.7,
                    help='Threshold')
args = parser.parse_args()

content=args.filename
segmentedImage_path=args.segmentedImage_path
content_path = args.content_path
content_image = utils.load_image2(content_path, img_size=args.img_size)
content_image = content_image.to(device)
content_features = utils.get_features(img_normalize(content_image), VGG)
target = content_image.clone().requires_grad_(True).to(device)
image_segmap_crf=np.load(segmentedImage_path, allow_pickle=True)

# print(content_image.shape)
#
# print(type(content_features))
# content_features.keys()

style_net = StyleNet.UNet()
style_net.to(device)

content_weight = args.lambda_c
crop_size = args.crop_size

optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
steps = args.max_step

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []
loss_patch_epoch=[]
reg_tv_epoch=[]
loss_glob_epoch=[]
loss_patch_context_epoch=[]


cropper = transforms.RandomCrop(args.crop_size)

augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(512)
])

source = "Night"
prompt = args.text

optimize = True # whether use optimized loss
people_scale = 0.3 # penalty of potrait
back_scale = 1.0 # penalty of back
window_width = 0.2 # portion of patch size to determine area
back_thres = 0.7 # thres rejection of potrait
people_thres = 0.7 # thres rejection of back
peo_num = 0.2 # portion of patch in potrait

object_Category=args.object_category
# print(object_Category)
mask = torch.tensor(np.repeat((image_segmap_crf.reshape(1, 1, 512, 512) == object_Category), 3, axis=1)).to(device)

# print(type(mask))
# print(mask.shape)

# def contextual_loss(x, y, h=0.5):
#     """Computes contextual loss between x and y.
#
#     Args:
#       x: features of shape (N, C, H, W).
#       y: features of shape (N, C, H, W).
#
#     Returns:
#       cx_loss = contextual loss between x and y (Eq (1) in the paper)
#     """
#     # cx_loss=100.0
#     # print(x.shape)    # 1 * 256
#     # print(y.shape)    # 1 * 256
#     assert x.size() == y.size()
#     N, F = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).
#     # print(N,F)
#     # y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
#     y_mu = y.mean(1).reshape(-1,1)
#     #
#     x_centered = x - y_mu
#     y_centered = y - y_mu
#     x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
#     y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)
#     # print(x_normalized.shape, y_normalized.shape)     # (1,256) and (1,256)
#     #
#     # # The equation at the bottom of page 6 in the paper
#     # # Vectorized computation of cosine similarity for each pair of x_i and y_j
#     # x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, F)
#     # y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, F)
#     x_normalized = x_normalized.reshape(N, -1)  # (N, F)
#     y_normalized = y_normalized.reshape(N, -1)  # (N, F)
#     # print(x_normalized.shape, y_normalized.shape)     #  (1,256) and (1,256)
#     # cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, F, F)
#
#     x_normalized_resize=x_normalized.resize(N, 1,F)
#     y_normalized_resize=y_normalized.resize(N, 1, F)
#     xT_normalized_resize = x_normalized_resize.transpose(1,2)
#     cosine_sim=torch.matmul(xT_normalized_resize, y_normalized_resize)
#     # print(cosine_sim.shape)    #(1,256,256)
#     d = 1 - cosine_sim  # (N, F, F)  d[n, i, j] means d_ij for n-th data
#     # d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, F, 1)
#     d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, F, 1)
#
#     # # Eq (2)
#     d_tilde = d / (d_min + 1e-5)
#
#     # # Eq(3)
#     w = torch.exp((1 - d_tilde) / h)
#     # print(w[0][0])   # values range = [0.94----0.98]
#     # print("shapes of d = (%d,%d), d_min = (%d,%d), d_tilde = (%d,%d)" %((d.shape), (d_min.shape),(d_tilde.shape)))
#     # print(d.shape, d_min.shape, d_tilde.shape)     #  (32,256,256) , (32,256,1), (32,256,256)
#     # # Eq(4)
#     # cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
#     cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
#     # print(cx_ij[0][0])   #value range #0.0038, 0.0039, 0.0040
#     # print("shapes of w = (%d,%d), cx_ij = (%d,%d)" %((w.shape),(cx_ij.shape)))
#     # print(w.shape, cx_ij.shape)   # (1,256,256) , (1,256,256)
#
#     # # Eq (1)
#     # cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
#
#     # print(type(cx))
#
#     cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, ) REmoved *10000
#     cx_loss = torch.mean(-torch.log(cx + 1e-5))
#     # print("shapes of cx = (%d,%d) and cx_loss= (%d,%d)" %((cx.shape), (cx_loss.shape)) )
#     # print(cx.shape, cx_loss.shape)       # (32) and ([])
#     # print("cx=%.2f and cx_loss=%.2f" %(cx, cx_loss) )
#     # print(cx, cx_loss)
#     return cx_loss

print(os.getcwd())
os.makedirs('results', mode = 0o777, exist_ok = True)
os.makedirs('outputs', mode = 0o777, exist_ok = True)


title= 'results/spectral_' + args.text + '_' + content + '.csv'
file = open(title, "w")
file.write("iterations, Total Loss, Content Loss, Patch Loss, Regularization TV Loss, Global Loss\n")


with torch.no_grad():
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)

    # new
    source_features1 = clip_model.encode_image(clip_normalize(content_image.masked_fill(mask, 0), device))
    source_features1 /= (source_features1.clone().norm(dim=-1, keepdim=True))
    # endnew
    source_features = clip_model.encode_image(clip_normalize(content_image, device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

num_crops = args.num_crops
img_size = args.img_size

for epoch in range(0, steps + 1):

    target = style_net(content_image, use_sigmoid=True).to(device)
    target.requires_grad_(True)

    target_features = utils.get_features(img_normalize(target), VGG)
    content_loss = 0

    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    seg = image_segmap_crf
    # new
    if optimize:
        back_proc, peo_proc, thres, scales = [], [], [], []
        while (len(back_proc) + len(peo_proc)) != args.num_crops:
            (i, j, h, w) = cropper.get_params(target, (crop_size, crop_size))
            target_crop = transforms.functional.crop(target, i, j, h, w)
            target_crop = augment(target_crop)
            if object_Category in seg[i + h - int(h * window_width):i + h, j:j + w]:  # potrait
                if len(peo_proc) < int(args.num_crops * peo_num):
                    peo_proc.append(target_crop)
                    scales.append(people_scale)
                    thres.append(people_thres)
            else:  # background
                back_proc.append(target_crop)
                scales.append(back_scale)
                thres.append(back_thres)
        img_proc = back_proc + peo_proc
    else:
        img_proc = []
        for i in range(args.num_crops):
            (i, j, h, w) = cropper.get_params(target, (crop_size, crop_size))
            target_crop = transforms.functional.crop(target, i, j, h, w)
            target_crop = augment(target_crop)
            img_proc.append(target_crop)
    # endnew
    img_proc = torch.cat(img_proc, dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug, device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

    img_direction = (image_features - source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))

    # # Contextual Loss
    # loss_patch_context = 0
    # # loss_patch_temp = contextual_loss(img_direction, text_direction)
    # # print(image_features.shape)
    # # print(text_features.shape)
    # # print(img_direction.shape)
    # # print(text_direction.shape)
    # loss_patch_temp = contextual_loss(image_features, text_direction)
    # loss_patch_temp = torch.abs(loss_patch_temp)
    # # loss_patch_temp[loss_patch_temp < args.thresh] = 0
    # loss_patch_context += loss_patch_temp.mean()  # #Context Patch Loss

    loss_patch = 0.0
    if optimize:
        for index, loss in enumerate(loss_temp):
            if loss >= thres[index]:
                loss_patch += loss * scales[index]
        loss_patch /= num_crops
    else:
        for index, loss in enumerate(loss_temp):
            if loss >= args.thresh:
                loss_patch += loss
        loss_patch /= num_crops

    if optimize:
        glob_features = clip_model.encode_image(clip_normalize(target.masked_fill(mask, 0), device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
        glob_direction = (glob_features - source_features1)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    else:
        glob_features = clip_model.encode_image(clip_normalize(target, device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
        glob_direction = (glob_features - source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

    loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    reg_tv = args.lambda_tv * get_image_prior_losses(target)
    total_loss = args.lambda_patch * loss_patch + content_weight * content_loss + reg_tv  + args.lambda_dir * loss_glob      ## + args.lambda_patch_context * loss_patch_context
    total_loss_epoch.append(total_loss.detach().cpu().numpy())
    loss_patch_epoch.append(loss_patch.detach().cpu().numpy())
    content_loss_epoch.append(content_loss.detach().cpu().numpy())
    reg_tv_epoch.append(reg_tv.detach().cpu().numpy())
    loss_glob_epoch.append(loss_glob.detach().cpu().numpy())
    # loss_patch_context_epoch.append(loss_patch_context.detach().cpu().numpy())

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print("After %d iters:" % epoch)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('patch loss: ', loss_patch.item())
        print('dir loss: ', loss_glob.item())
        print('TV loss: ', reg_tv.item())
        file.write(str(epoch) + "," + str(total_loss.item()) + "," + str(content_loss.item()) + "," + str(
            loss_patch.item()) + "," + str(reg_tv.item()) + "," + str(loss_glob.item()) + "\n")

        # print('Patch Context Loss', loss_patch_context.item())
    if epoch % 100 == 0:
        out_path = 'outputs/spectral_' + args.text + '_' + content + '_' + args.exp_name + '.jpg'
        output_image = target.clone()
        output_image = torch.clamp(output_image, 0, 1)
        output_image = adjust_contrast(output_image, 1.5)
        # plt.figure(figsize=(5, 5))
        # plt.imshow(utils.im_convert2(output_image))
        # plt.show()
        vutils.save_image(output_image, out_path, nrow=1, normalize=True)
torch.cuda.empty_cache()
# print(output_image.shape)
file.close()


# epochs = range(0, 401)
# title='ClipStyler_Spectral_TotalLoss'
#
# plt.plot(epochs, total_loss_epoch, label='Total Loss')
# plt.title('All Losses')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.xticks(np.arange(0, 450, 50))
# # plt.legend(loc='')
# plt.show()
# # save image
# plt.savefig(title + ".png")  # should before show method
