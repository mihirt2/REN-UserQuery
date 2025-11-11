# Testing example:
#
# test_local.py  (copy of your test.py with small edits)
import yaml, argparse
from pathlib import Path
from PIL import Image
import torch, torchvision.transforms as T
from ren import REN, XREN

def load_image(path, size):
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tfm(img).unsqueeze(0)

def test_ren_local():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_extractor', required=True, help='e.g. dinov2_vitl14')
    parser.add_argument('--image', required=True, help='path to local image')
    parser.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    args = parser.parse_args()

    device = (
        'cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()))
        else 'cpu'
    )

    with open(f'configs/ren_{args.feature_extractor}.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    size = config['parameters']['image_resolution']

    # Load local image
    image = load_image(args.image, size).to(device)

    # Load REN
    ren = REN(config)  # make sure REN/XREN internally respect device; if not, pass device in ctor
    ren = ren.to(device).eval()

    # Extract region tokens
    with torch.no_grad():
        region_tokens = ren(image)
    print('Region tokens shape:', region_tokens[0].shape)

if __name__ == '__main__':
    test_ren_local()
