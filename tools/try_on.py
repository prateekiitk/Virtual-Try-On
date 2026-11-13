"""Virtual try-on on your own photos, in one command.

    .venv\\Scripts\\python tools\\try_on.py --person "path\\to\\person.jpg" --garment "path\\to\\shirt.jpg"

What it does:
  1. finds 17 body keypoints on the person (torchvision Keypoint R-CNN)
  2. crops the photo to the VITON framing (head to thighs, 192 x 256)
  3. labels hair / face / top / arms / trousers (SegFormer clothing model -> ACGPN's 14 classes)
  4. cuts the garment out of its background and fits it to 192 x 256
  5. runs the pretrained ACGPN model (ACGPN_inference/checkpoints/label2city)
  6. saves the result in my_results/

Photo tips: one person, facing the camera, standing, head to at least the hips visible, plain background.
Garment tips: a flat product photo of a top on a plain (white or grey) background.
The first run downloads two helper models (about 400 MB) and needs internet; later runs work offline.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np, cv2, torch, torchvision
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 192, 256

ap = argparse.ArgumentParser(description="ACGPN virtual try-on on your own photos")
ap.add_argument("--person", required=True, help="photo of the person")
ap.add_argument("--garment", required=True, help="product photo of the top")
ap.add_argument("--out", default=os.path.join(ROOT, "my_results"), help="folder for the results")
args = ap.parse_args()
for p in (args.person, args.garment):
    if not os.path.isfile(p):
        sys.exit(f"File not found: {p}")
if not torch.cuda.is_available():
    sys.exit("No NVIDIA GPU found by PyTorch. Use the project's .venv (see README) on a computer with an NVIDIA GPU.")
dev = "cuda"

work = os.path.join(ROOT, "my_tryon_work")
shutil.rmtree(work, ignore_errors=True)
for d in ["img", "label", "pose", "color", "edge", "mask", "colormask"]:
    os.makedirs(os.path.join(work, "test_" + d))

def read_rgb(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)   # works with non-English paths too
    if im is None:
        sys.exit(f"Could not read image: {p}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# ---------------- 1. pose on the full photo ----------------
print("[1/5] Finding the body pose ...")
kp_net = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT").to(dev).eval()
full = read_rgb(args.person)
with torch.no_grad():
    o = kp_net([torch.from_numpy(full).permute(2, 0, 1).float().div(255).to(dev)])[0]
if len(o["scores"]) == 0 or float(o["scores"].max()) < 0.5:
    sys.exit("No person found in the photo. Use a clear photo of one person facing the camera.")
i = int(o["scores"].argmax())
kp, ks = o["keypoints"][i].cpu().numpy(), o["keypoints_scores"][i].cpu().numpy()
if min(ks[5], ks[6]) < 0:
    sys.exit("Both shoulders must be visible in the photo.")
neck = (kp[5, :2] + kp[6, :2]) / 2

# ---------------- 2. crop to VITON framing ----------------
print("[2/5] Cropping the photo to the model's framing ...")
# typical VITON positions (median over the test set): neck y, hip y, shoulder width
T_NECK, T_HIP, T_SHOULDER = 82.6, 197.1, 77.9
hips_ok = min(ks[11], ks[12]) > 0
if hips_ok:
    scale = (T_HIP - T_NECK) / ((kp[11, 1] + kp[12, 1]) / 2 - neck[1])
else:  # half-body photo: use shoulder width instead
    scale = T_SHOULDER / abs(kp[5, 0] - kp[6, 0])
x0, y0 = neck[0] - (W / 2) / scale, neck[1] - T_NECK / scale
M = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
crop = cv2.warpAffine(full, M, (W, H), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
crop4 = cv2.warpAffine(full, M * np.float32([[4], [4]]), (W * 4, H * 4), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
cv2.imwrite(os.path.join(work, "test_img", "900001_0.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

kpc = np.c_[kp[:, :2], np.ones(17)] @ M.T
pose = []
for c in [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]:   # COCO -> OpenPose-18 order
    if c is None:
        (x, y), conf = (kpc[5] + kpc[6]) / 2, min(ks[5], ks[6])
    else:
        (x, y), conf = kpc[c], ks[c]
    pose += [float(x), float(y), 0.9] if (1 < x < W - 1 and 1 < y < H - 1 and conf > 0) else [0.0, 0.0, 0.0]
json.dump({"version": 1.0, "people": [{"face_keypoints": [], "pose_keypoints": pose}]},
          open(os.path.join(work, "test_pose", "900001_0_keypoints.json"), "w"))

# ---------------- 3. body parts ----------------
print("[3/5] Labelling hair, face, clothes and arms ...")
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
name = "mattmdjaga/segformer_b2_clothes"
proc = SegformerImageProcessor.from_pretrained(name)
seg = AutoModelForSemanticSegmentation.from_pretrained(name).to(dev).eval()
with torch.no_grad():
    logits = seg(**proc(images=Image.fromarray(crop4), return_tensors="pt").to(dev)).logits
    atr = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
to_acgpn = {0: 0, 1: 1, 2: 1, 3: 12, 4: 4, 5: 8, 6: 8, 7: 4, 8: 8, 9: 5, 10: 6, 11: 12, 12: 9, 13: 9, 14: 11, 15: 11, 16: 0, 17: 4}
lab = np.vectorize(to_acgpn.get)(atr).astype(np.uint8)
cx = (kpc[5, 0] + kpc[6, 0]) / 2
xs = np.arange(W)[None, :].repeat(H, 0)
lab[(lab == 11) & (xs < cx)] = 13          # VITON: arm 11 on the image's right, 13 on the left
lab[(lab == 9) & (xs < cx)] = 10
if not (lab == 4).any():
    print("   Note: no top/shirt was found on the person; results may be poor.")
cv2.imwrite(os.path.join(work, "test_label", "900001_0.png"), lab)

# ---------------- 4. garment cut-out ----------------
print("[4/5] Cutting the garment out of its background ...")
g = read_rgb(args.garment)
border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]).astype(np.float32)
bg = np.median(border, axis=0)
fg = (np.abs(g.astype(np.float32) - bg).sum(2) > 40).astype(np.uint8) * 255
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    sys.exit("Could not find the garment. Use a product photo on a plain background.")
big = max(cnts, key=cv2.contourArea)
mask = np.zeros_like(fg); cv2.drawContours(mask, [big], -1, 255, -1)   # filled outline
x, y, w, h = cv2.boundingRect(big)
g_c, m_c = g[y:y + h, x:x + w], mask[y:y + h, x:x + w]
s = min(W * 0.92 / w, H * 0.94 / h)                                     # VITON garments nearly fill the frame
nw, nh = max(1, int(w * s)), max(1, int(h * s))
g_r = cv2.resize(g_c, (nw, nh), interpolation=cv2.INTER_AREA)
m_r = cv2.resize(m_c, (nw, nh), interpolation=cv2.INTER_NEAREST)
color = np.full((H, W, 3), 255, np.uint8); edge = np.zeros((H, W), np.uint8)
ox, oy = (W - nw) // 2, (H - nh) // 2
color[oy:oy + nh, ox:ox + nw] = np.where(m_r[..., None] > 0, g_r, 255)
edge[oy:oy + nh, ox:ox + nw] = m_r
cv2.imwrite(os.path.join(work, "test_color", "900001_1.jpg"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(os.path.join(work, "test_edge", "900001_1.jpg"), edge)
for d in ["mask", "colormask"]:
    cv2.imwrite(os.path.join(work, "test_" + d, "900001_0.png"), np.zeros((H, W), np.uint8))
del kp_net, seg
torch.cuda.empty_cache()

# ---------------- 5. run ACGPN ----------------
print("[5/5] Running the try-on model (about 30-60 seconds) ...")
inf = os.path.join(ROOT, "ACGPN_inference")
env = dict(os.environ, ACGPN_RECON="1")
# test.py loads the weights from ./checkpoints, so it must run inside ACGPN_inference;
# it writes its output to ACGPN_inference/sample/900001_0.jpg, which we move out afterwards
raw_tmp = os.path.join(inf, "sample", "900001_0.jpg")
if os.path.exists(raw_tmp):
    os.remove(raw_tmp)
r = subprocess.run([sys.executable, "test.py", "--nThreads", "0", "--serial_batches", "--dataroot", work + os.sep],
                   cwd=inf, env=env, capture_output=True, text=True)
if r.returncode != 0 or not os.path.isfile(raw_tmp):
    print(r.stdout[-2000:], r.stderr[-2000:])
    sys.exit("The model failed; see the messages above.")
raw = os.path.join(work, "result_strip.jpg")
shutil.move(raw_tmp, raw)

# ---------------- save ----------------
os.makedirs(args.out, exist_ok=True)
base = os.path.splitext(os.path.basename(args.person))[0] + "__" + os.path.splitext(os.path.basename(args.garment))[0]
strip = cv2.imread(raw)
tiles = [("your photo", strip[:, 768:960]), ("garment", strip[:, 384:576]), ("try-on result", strip[:, 576:768])]
row = []
for t, im in tiles:
    im = cv2.resize(im, (288, 384), interpolation=cv2.INTER_CUBIC)
    bar = np.full((34, 288, 3), 255, np.uint8)
    cv2.putText(bar, t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    row += [np.vstack([bar, im]), np.full((418, 8, 3), 255, np.uint8)]
out_cmp = os.path.join(args.out, base + "_comparison.png")
out_res = os.path.join(args.out, base + "_result.png")
cv2.imwrite(out_cmp, np.hstack(row[:-1]))
cv2.imwrite(out_res, cv2.resize(strip[:, 576:768], (384, 512), interpolation=cv2.INTER_CUBIC))
print("\nDone!")
print("  Result:     ", out_res)
print("  Comparison: ", out_cmp)
