import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

import StyleNet
import utils
import clip
import torch.nn.functional as F
from template import imagenet_templates

import argparse
from torchvision.transforms.functional import adjust_contrast
from torchvision import utils as vutils
# from extract import extract_utils as eutils

from IPython.display import display
from argparse import Namespace
import os

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
VGG = models.vgg19(pretrained=False).features
VGG.to(device)
for parameter in VGG.parameters():
    parameter.requires_grad_(False)

# training_args = {
#     "lambda_tv": 2e-3,
#     "lambda_patch": 9000,
#     "lambda_dir_b": 500,
#     "lambda_dir_f": 500,
#     "lambda_c": 600,
#     "crop_size": 128,
#     "num_crops": 64,
#     "img_size": 512,
#     "max_step": 400,
#     "lr": 5e-4,
#     "thresh": 0.7,
#     "content_path": "../test_set/images/2009_000486_resized.png",
#     "segmentedImage_path":"../segmaps/deep_spectral/single_region_segmentation/crf_deep_spectral/2009_000486_resized.npy",
#     "text": "Starry Night by Vincent van gogh",
#     "textb": "Green crystals",
#     "filename":"Bird",
#     "object_category":255
#     "clip_model":"RN50"
# }
# args = Namespace(**training_args)

root="/home/phdcs2/Hard_Disk/Outputs/WACV/"
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, default="2009_000486_resized",
                    help='Image resolution')
parser.add_argument('--content_path', type=str, default="test_set/images/2009_000486_resized.png",
                    help='Image resolution')
parser.add_argument('--img_size', type=str, default=512,
                    help='Image size')
parser.add_argument('--segmentedImage_path', type=str, default="segmaps/deep_spectral/single_region_segmentation/crf_deep_spectral/2009_000486_resized.npy",
                    help='Segmented Image Path')
parser.add_argument('--content_name', type=str, default="face",
                    help='Content Name')
parser.add_argument('--exp_name', type=str, default="exp1",
                    help='Exepriment Name')
parser.add_argument('--object_category', type=int, default=255,
                    help='Object Category')
parser.add_argument('--clip_model', type=str, default="RN50",
                    help='CLIP model Name')
parser.add_argument('--textf', type=str, default="Golden",
                    help='Text Description')
parser.add_argument('--textb', type=str, default="Starry Night by Vincent van gogh",
                    help='Text Description')
parser.add_argument('--lambda_tv', type=float, default=2e-3,
                    help='total variation loss parameter')
parser.add_argument('--lambda_patch', type=float, default=9000,
                    help='PatchCLIP loss parameter')
parser.add_argument('--lambda_dir_b', type=float, default=500,
                    help='directional loss parameter')
parser.add_argument('--lambda_dir_f', type=float, default=500,
                    help='directional loss parameter')
parser.add_argument('--lambda_c', type=float, default=600,
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

filename=args.filename
contentname=args.content_name
# torch.cuda.empty_cache()
segmentedImage_path=args.segmentedImage_path
content_path = args.content_path
content_image = utils.load_image2(content_path, img_size=args.img_size)
content_image = content_image.to(device)
content_features = utils.get_features(img_normalize(content_image), VGG)
target = content_image.clone().requires_grad_(True).to(device)
seg=np.load(segmentedImage_path, allow_pickle=True)

# plt.figure(figsize=(4, 4))
# plt.imshow(seg)
# plt.show()

# image = utils.load_image2(args.content_path, img_size=512)


# content=args.filename
# content_path = args.content_path
# content_image = utils.load_image2(content_path, img_size=args.img_size)
# content_image = content_image.to(device)
# content_features = utils.get_features(img_normalize(content_image), VGG)
# target = content_image.clone().requires_grad_(True).to(device)
# print(clip.available_models())
# clip_model, preprocess = clip.load(args.clip_model, device, jit=False)
clip_model, preprocess = clip.load(args.clip_model, device, jit=False)

style_net = StyleNet.UNet()
style_net.to(device)

content_weight = args.lambda_c
crop_size = args.crop_size
# crop_size=128

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
    transforms.Resize(224)
])

source = "a Photo"
promptf = args.textb
promptb = args.textf

optimize = True # whether use optimized loss
people_scale = 0.3 # penalty of potrait
back_scale = 1.0 # penalty of back
window_width = 0.2 # portion of patch size to determine area
back_thres = 0.7 # thres rejection of potrait
people_thres = 0.7 # thres rejection of back
peo_num = 0.2 # portion of patch in potrait

object_category=args.object_category
# content=args.filename
num_crops = args.num_crops
img_size = args.img_size


mask = torch.tensor(np.repeat((seg.reshape(1, 1, 512, 512) == object_category), 3, axis=1)).to(device)
maskb = torch.tensor(np.repeat((seg.reshape(1, 1, 512, 512) != object_category), 3, axis=1)).to(device)

# # title= 'new_res/spectral_' + promptb + '_' + promptf + '_' +  content + '.csv'
# title= 'results/spectral_globfb' + promptb + '_' + promptf + '_' +  content + '.csv'
# file = open(title, "w")
# file.write("iterations, Total Loss, Content Loss, Patch Loss, Regularization TV Loss, Global Background  Loss, Global Foregound Loss\n")

results=root + 'Methods/' + args.exp_name + '/results/'
isExist = os.path.exists(results)
if not isExist:
    eutils.make_output_dir(results)
res_title = results + args.exp_name + '_'+ promptb + '_' + promptf + '_' + filename + '_' + contentname + '.csv'


outputs = root + 'Methods/' + args.exp_name + '/' + 'outputs/'
isExist = os.path.exists(outputs)
if not isExist:
    eutils.make_output_dir(outputs)
output_title = outputs

file = open(res_title, "w")
file.write("iterations, Total Loss, Content Loss, Patch Loss, Regularization TV Loss, Global Background  Loss, Global Foregound Loss\n")


def softmax3d(input):
    m = nn.Softmax()
    a, b = input.size()
    # print(a)
    # print()
    input = torch.reshape(input, (1, -1))
    output = m(input)
    output = torch.reshape(output, (a, b))
    return output

with torch.no_grad():

    template_text = compose_text_with_templates(promptf, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    template_text = compose_text_with_templates(promptb, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_featuresb = clip_model.encode_text(tokens).detach()
    text_featuresb = text_featuresb.mean(axis=0, keepdim=True)
    text_featuresb /= text_featuresb.norm(dim=-1, keepdim=True)

    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)

    # source_features1 = clip_model.encode_image(clip_normalize(content_image.masked_fill(mask, 0), device))
    # source_features1 /= (source_features1.clone().norm(dim=-1, keepdim=True))
    # temp_sf1=source_features1
    # source_features1=softmax3d(temp_sf1)

    source_features_b = clip_model.encode_image(clip_normalize(content_image.masked_fill(mask, 0), device))
    source_features_b /= (source_features_b.clone().norm(dim=-1, keepdim=True))
    temp_sf1=source_features_b
    source_features_b=softmax3d(temp_sf1)

    source_features_f = clip_model.encode_image(clip_normalize(content_image.masked_fill(maskb, 0), device))
    source_features_f /= (source_features_f.clone().norm(dim=-1, keepdim=True))
    temp_sf1=source_features_f
    source_features_f=softmax3d(temp_sf1)

    source_features = clip_model.encode_image(clip_normalize(content_image, device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
    temp_sf=source_features
    source_features=softmax3d(temp_sf)

for epoch in range(0, steps+1):

    target = style_net(content_image,use_sigmoid=True).to(device)
    target.requires_grad_(True)

    target_features = utils.get_features(img_normalize(target), VGG)
    content_loss = 0

    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    if optimize:
        back_proc, peo_proc, thres, scales =[], [], [], []
        while (len(back_proc) + len(peo_proc)) != args.num_crops:
            (i, j, h, w) = cropper.get_params(target, (crop_size, crop_size))
            target_crop = transforms.functional.crop(target, i, j, h, w)
            target_crop = augment(target_crop)
            if object_category in seg[i+h-int(h * window_width):i+h, j:j+w]: # potrait
                if len(peo_proc) < int(args.num_crops * peo_num):
                    peo_proc.append(target_crop)
                    scales.append(people_scale)
                    thres.append(people_thres)
            else: # background
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

    img_proc = torch.cat(img_proc,dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)

    text_directionb = (text_featuresb-text_source).repeat(image_features.size(0),1)
    text_directionb /= text_directionb.norm(dim=-1, keepdim=True)

    loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))

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
        glob_features_maskb = clip_model.encode_image(clip_normalize(target.masked_fill(mask, 0),device))
        glob_features_maskb /= (glob_features_maskb.clone().norm(dim=-1, keepdim=True))
        # temp_sf1=glob_features_maskb
        # glob_features_maskb=softmax3d(temp_sf1)


        glob_features_maskf = clip_model.encode_image(clip_normalize(target.masked_fill(maskb, 0),device))
        glob_features_maskf /= (glob_features_maskf.clone().norm(dim=-1, keepdim=True))
        # temp_sf1=glob_features_maskf
        # glob_features_maskf=softmax3d(temp_sf1)

        glob_directionf = (glob_features_maskf-source_features_f)
        glob_directionf /= glob_directionf.clone().norm(dim=-1, keepdim=True)

        glob_directionb = (glob_features_maskb-source_features_b)
        glob_directionb /= glob_directionb.clone().norm(dim=-1, keepdim=True)

    else:
        glob_features = clip_model.encode_image(clip_normalize(target,device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
        glob_direction = (glob_features-source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

    loss_glob_b = (1 - torch.cosine_similarity(glob_directionb, text_directionb, dim=1)).mean()
    loss_glob_f = (1 - torch.cosine_similarity(glob_directionf, text_direction, dim=1)).mean()

    reg_tv = args.lambda_tv * get_image_prior_losses(target)
    total_loss = args.lambda_patch * loss_patch + content_weight * content_loss + reg_tv + args.lambda_dir_b * loss_glob_b + args.lambda_dir_f * loss_glob_f
    total_loss_epoch.append(total_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print("After %d iters:" % epoch)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('patch loss: ', loss_patch.item())
        print('Background dir loss: ', loss_glob_b.item())
        print('Foreground dir loss: ', loss_glob_f.item())
        print('TV loss: ', reg_tv.item())
        file.write(str(epoch) + "," + str(total_loss.item()) + "," + str(content_loss.item()) + "," + str(
            loss_patch.item()) + "," + str(reg_tv.item()) + "," + str(loss_glob_b.item()) + "," + str(
            loss_glob_f.item()) + "\n")

    if epoch % 100 ==0:
        # out_path = 'new_loss_ops/' + 'spectral_globfb' + promptb + '_' + promptf + content  + '.jpg'
        out_path = output_title + args.exp_name  + 'spectral_globfb' + promptb + '_' + promptf +'_' + filename + '_' + contentname  + '.jpg'
        output_image = target.clone()
        output_image = torch.clamp(output_image,0,1)
        output_image = adjust_contrast(output_image,1.5)
        vutils.save_image(output_image, out_path, nrow=1, normalize=True)
        # plt.figure(figsize=(4,4))
        # plt.imshow(utils.im_convert2(output_image))
        # plt.show()

torch.cuda.empty_cache()
file.close()