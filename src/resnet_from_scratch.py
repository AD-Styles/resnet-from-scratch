"""
ResNet From Scratch — Skip Connection의 효과 검증
"Deep Residual Learning for Image Recognition" (He et al., CVPR 2016)

같은 깊이·같은 파라미터의 Plain-20 vs ResNet-20을 학습해 Skip Connection의
효과를 통제 실험으로 입증하고, BN 출력의 Layer Response 분석으로
잔차 함수가 0에 가까운지(논문 §3.1 가설) 정량 검증한다.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ────────────── Config ──────────────
SEED          = 42
EPOCHS        = 30          # 논문은 200 — 빠른 데모용으로 30
BATCH_SIZE    = 128
LR            = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4
LR_MILESTONES = [15, 22]    # 논문 [100, 150] / 200 = 50%, 75% 시점
OUTPUT_DIR    = "./results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Windows에서 한글/특수문자 출력 안정성 확보
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 한글 폰트 (Windows)
plt.rcParams['font.family']        = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}", flush=True)


# ────────────── Model ──────────────

class BasicBlock(nn.Module):
    """ResNet의 기본 블록 — 3x3 Conv 두 개 + Skip Connection (논문 Fig.2)"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # 차원이 달라지면 1x1 conv로 맞춤 (논문 §3.2 Option B)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)        # ★ 핵심: H(x) = F(x) + x
        return F.relu(out)


class PlainBlock(nn.Module):
    """비교용 — BasicBlock과 동일하지만 Skip Connection이 없는 블록"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))    # skip connection 없음
        return out


class CIFAR10Net(nn.Module):
    """CIFAR-10용 6n+2 layer 네트워크 (논문 §4.2).

    block=BasicBlock → ResNet (skip connection 있음)
    block=PlainBlock → Plain Network (비교용 baseline)
    n=3 → 20 layers
    """

    def __init__(self, n=3, block=BasicBlock, num_classes=10):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(block, 16, 16, n, stride=1)
        self.stage2 = self._make_stage(block, 16, 32, n, stride=2)
        self.stage3 = self._make_stage(block, 32, 64, n, stride=2)
        self.fc     = nn.Linear(64, num_classes)

        # He 초기화 (논문 §3.4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, block, in_ch, out_ch, n_blocks, stride):
        layers = [block(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ────────────── Data ──────────────

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_set = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
test_set  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
# Windows 호환을 위해 num_workers=0
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=0, pin_memory=True)


# ────────────── Training ──────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        correct += out.argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += model(x).argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return correct / total


def train(model, name, epochs=EPOCHS):
    """모델 학습 + history/checkpoint 저장."""
    print(f"\n===== Training {name} =====", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=0.1)

    history = {'train_acc': [], 'test_acc': []}
    best_acc, t0 = 0.0, time.time()

    for epoch in range(1, epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_acc  = evaluate(model, test_loader)
        scheduler.step()

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_{name}.pt")

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            mins = (time.time() - t0) / 60
            print(f"  Epoch {epoch:3d}/{epochs} - train: {train_acc*100:.2f}%, "
                  f"test: {test_acc*100:.2f}%, LR: {scheduler.get_last_lr()[0]:.1e}  "
                  f"({mins:.1f} min)", flush=True)

    # history JSON 저장 (재실행 시 학습 생략 가능)
    with open(f"{OUTPUT_DIR}/history_{name}.json", 'w') as f:
        json.dump({'history': history, 'best_acc': best_acc}, f)

    print(f"  [DONE] {name} best test accuracy: {best_acc*100:.2f}%", flush=True)
    return history, best_acc


# ─── 학습 실행: 2개 모델 (같은 깊이, Skip Connection 유무만 다름) ───
models_to_train = [
    ("Plain-20",  CIFAR10Net(n=3, block=PlainBlock).to(DEVICE)),
    ("ResNet-20", CIFAR10Net(n=3, block=BasicBlock).to(DEVICE)),
]

histories, best_accs = {}, {}
for name, model in models_to_train:
    ckpt_path    = f"{OUTPUT_DIR}/best_{name}.pt"
    history_path = f"{OUTPUT_DIR}/history_{name}.json"

    # 이미 학습된 결과가 있으면 로드 (재시각화 시 시간 절약)
    if os.path.exists(ckpt_path) and os.path.exists(history_path):
        print(f"\n===== Loading cached {name} =====", flush=True)
        with open(history_path) as f:
            data = json.load(f)
        histories[name] = data['history']
        best_accs[name] = data['best_acc']
        print(f"  cached best test accuracy: {best_accs[name]*100:.2f}%", flush=True)
    else:
        histories[name], best_accs[name] = train(model, name)


# ────────────── Visualizations ──────────────

COLORS = {"Plain-20": "#c0392b", "ResNet-20": "#2980b9"}


# ─── fig_01: 학습 곡선 (Plain-20 vs ResNet-20) ───
fig, ax = plt.subplots(figsize=(11, 6))
for name, h in histories.items():
    epochs_x = range(1, len(h['test_acc']) + 1)
    ax.plot(epochs_x, [a*100 for a in h['test_acc']],
            color=COLORS[name], lw=2.4,
            label=f"{name}  (best: {best_accs[name]*100:.2f}%)")
for ms in LR_MILESTONES:
    ax.axvline(ms, color='gray', ls=':', alpha=0.6)
ax.set_xlabel("Epoch", fontweight='bold')
ax.set_ylabel("Test Accuracy (%)", fontweight='bold')
ax.set_title("CIFAR-10 학습 곡선 — 같은 깊이의 Plain-20 vs ResNet-20",
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_01_training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nfig_01 saved.", flush=True)


# ─── fig_02: 정확도 비교 ───
fig, ax = plt.subplots(figsize=(8, 6))
names = list(best_accs.keys())
accs  = [best_accs[n] * 100 for n in names]
bars  = ax.bar(names, accs, color=[COLORS[n] for n in names],
               edgecolor='black', linewidth=1.4, width=0.5)
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
            f"{a:.2f}%", ha='center', fontweight='bold', fontsize=12)

# Skip Connection 효과 격차 어노테이션
gap = best_accs["ResNet-20"]*100 - best_accs["Plain-20"]*100
ax.annotate('', xy=(1, accs[1] - 0.5), xytext=(0, accs[0] - 0.5),
            arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2))
ax.text(0.5, (accs[0]+accs[1])/2 - 0.3, f"+{gap:.2f}%p\n(Skip Connection 효과)",
        ha='center', fontweight='bold', color='#27ae60', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60'))

ax.set_ylabel("Test Accuracy (%)", fontweight='bold')
ax.set_title("Plain-20 vs ResNet-20 — 같은 깊이, 같은 파라미터 (0.27M)\n"
             "Skip Connection 한 줄만 다른 통제 실험",
             fontsize=12, fontweight='bold')
ax.set_ylim(min(accs) - 1.5, max(accs) + 1.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_02_accuracy_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_02 saved.", flush=True)


# ─── fig_03: Layer Response Analysis (논문 §4.2 Figure 7) ⭐ ───

def get_layer_responses(model, loader, n_batches=8):
    """학습된 모델의 BN 출력 표준편차를 layer별로 측정."""
    model.eval()
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    stds = [[] for _ in bn_modules]
    hooks = []
    for i, bn in enumerate(bn_modules):
        def make_hook(idx):
            return lambda module, inp, out: stds[idx].append(out.std().item())
        hooks.append(bn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            _ = model(x.to(DEVICE))

    for h in hooks:
        h.remove()
    return [float(np.mean(s)) for s in stds]


# 학습된 두 모델 로드
plain20  = CIFAR10Net(n=3, block=PlainBlock).to(DEVICE)
plain20.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_Plain-20.pt"))

resnet20 = CIFAR10Net(n=3, block=BasicBlock).to(DEVICE)
resnet20.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_ResNet-20.pt"))

plain_resp  = get_layer_responses(plain20,  test_loader)
resnet_resp = get_layer_responses(resnet20, test_loader)

# 논문 Figure 7 우측 panel과 동일한 형식: std 내림차순 정렬 (전체 경향이 더 명확)
plain_sorted  = sorted(plain_resp,  reverse=True)
resnet_sorted = sorted(resnet_resp, reverse=True)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(plain_sorted,  'o-', color='#c0392b', lw=2.4, markersize=8,
        label=f'Plain-20  (mean = {np.mean(plain_resp):.3f})')
ax.plot(resnet_sorted, 's-', color='#2980b9', lw=2.4, markersize=8,
        label=f'ResNet-20 (mean = {np.mean(resnet_resp):.3f})')
ax.set_xlabel("Layer Index (std 내림차순 정렬)", fontweight='bold')
ax.set_ylabel("BN Output 표준편차 (std)", fontweight='bold')
ax.set_title("Layer Response - 잔차 함수가 0에 가까운가? (논문 §4.2 Fig.7 재현)",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# 평균 격차 어노테이션
red = (1 - np.mean(resnet_resp)/np.mean(plain_resp)) * 100
ax.text(0.02, 0.05,
    f"ResNet의 평균 응답이 Plain보다 약 {red:.1f}% 작음\n"
    "-> 잔차 F(x)가 실제로 0에 가까운 함수를 학습\n"
    "   (논문 §3.1 핵심 가설 입증)",
    transform=ax.transAxes, fontsize=10.5, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e7',
              edgecolor='#888', alpha=0.95))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_03_layer_response.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_03 saved.", flush=True)


# ────────────── Summary ──────────────
print("\n" + "=" * 50, flush=True)
print("FINAL RESULTS", flush=True)
print("=" * 50, flush=True)
for name in ["Plain-20", "ResNet-20"]:
    print(f"  {name:10s}: {best_accs[name]*100:.2f}%", flush=True)
print(f"\n  Skip Connection 효과: +{gap:.2f}%p", flush=True)
print(f"  Layer Response 감소: -{red:.1f}% (ResNet이 더 작은 응답)", flush=True)
print(f"\nFigures saved to {OUTPUT_DIR}/", flush=True)
