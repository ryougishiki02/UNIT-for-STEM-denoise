#  python test_batch.py --trainer UNIT --config configs/unit_STM_folder.yaml --input_folder inputs/batch_folder --output_folder results/batch_results --checkpoint models/unit_STM.pt --a2b 0
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
#from torch.autograd import Variable
from data import ImageFolder
import torchvision.utils as vutils
import sys
import torch
import os

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style', type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

opts = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size_a'], crop=False, num_workers=0)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

# Load checkpoint weights
try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint))
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()

# Select encode/decode function based on a2b flag
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode

# Start testing
with torch.no_grad():  # Use torch.no_grad() instead of volatile=True
    if opts.trainer == 'MUNIT':
        style_fixed = torch.randn(opts.num_style, style_dim, 1, 1).cuda()
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            print(names[1])
            images = images.cuda()
            content, _ = encode(images)
            style = style_fixed if opts.synchronized else torch.randn(opts.num_style, style_dim, 1, 1).cuda()
            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.0
                basename = os.path.basename(names[1])
                path = os.path.join(opts.output_folder + "_%02d" % j, basename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if not opts.output_only:
                vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)

    elif opts.trainer == 'UNIT':
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            print(names[1])
            images = images.cuda()
            content, _ = encode(images)
            outputs = decode(content)
            outputs = (outputs + 1) / 2.0
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder, basename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if not opts.output_only:
                vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)

    else:
        pass
