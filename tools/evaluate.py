"""SSIM and Inception Score for ACGPN reconstruction results, following the project README.

SSIM: same algorithm as Po-Hsun-Su/pytorch-ssim (Gaussian window 11, sigma 1.5, images in [0,1]).
IS:   same procedure as sbarratt/inception-score-pytorch (Inception-v3, inputs in [-1,1], resized to 299, splits=1).

usage: python evaluate.py <results_dir> <ground_truth_img_dir>
The result images are ACGPN's 5-panel strips; panel 4 (x 576..768) is the generated try-on.
"""
import os, sys, math
import numpy as np, cv2, torch, torch.nn.functional as F, torchvision

res_dir, gt_dir = sys.argv[1], sys.argv[2]
dev = 'cuda'

def gaussian_window(size=11, sigma=1.5, ch=3):
    g = torch.tensor([math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)])
    g = (g / g.sum()).unsqueeze(1)
    w = (g @ g.t()).float().unsqueeze(0).unsqueeze(0)
    return w.expand(ch, 1, size, size).contiguous()

WIN = gaussian_window().to(dev)

def ssim(a, b):  # a, b: 1x3xHxW in [0,1]
    p = 11 // 2
    mu1 = F.conv2d(a, WIN, padding=p, groups=3); mu2 = F.conv2d(b, WIN, padding=p, groups=3)
    s1 = F.conv2d(a * a, WIN, padding=p, groups=3) - mu1 ** 2
    s2 = F.conv2d(b * b, WIN, padding=p, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, WIN, padding=p, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return m.mean().item()

def to_tensor(bgr):
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0

names = sorted(os.listdir(res_dir))
ssims, fakes = [], []
for n in names:
    fake = cv2.imread(os.path.join(res_dir, n))[:, 576:768]
    real = cv2.imread(os.path.join(gt_dir, n))
    f, r = to_tensor(fake), to_tensor(real)
    ssims.append(ssim(f, r))
    fakes.append(f.cpu())
print(f"images: {len(names)}")
print(f"SSIM (generated vs real photo): {np.mean(ssims):.4f}")

net = torchvision.models.inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(dev).eval()
preds = []
with torch.no_grad():
    for i in range(0, len(fakes), 32):
        x = torch.cat(fakes[i:i + 32]).to(dev) * 2 - 1
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        preds.append(F.softmax(net(x), dim=1).cpu().numpy())
p = np.concatenate(preds)
py = p.mean(axis=0, keepdims=True)
kl = (p * (np.log(p + 1e-12) - np.log(py + 1e-12))).sum(axis=1)
print(f"Inception Score (splits=1): {np.exp(kl.mean()):.4f}")
