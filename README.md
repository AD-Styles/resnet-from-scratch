# 🧠 ResNet From Scratch — Skip Connection의 효과 검증
### "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016) 핵심 가설을 PyTorch로 재현

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-FF6B6B?style=for-the-badge)
![Paper](https://img.shields.io/badge/Paper-CVPR_2016-1976D2?style=for-the-badge)

---

## 📌 프로젝트 요약 (Project Overview)

ResNet 논문의 핵심 메시지는 한 줄로 요약됩니다 — *"네트워크가 깊어질수록 학습이 어려워지는 문제(Degradation Problem)를, 잔차 연결(`H(x) = F(x) + x`) 한 줄로 해결한다."* 이 포트폴리오는 그 주장을 두 가지 실험으로 직접 검증합니다.

**1. 같은 조건에서 Skip Connection 한 줄만 빼면 어떻게 되는가?**
Plain-20과 ResNet-20은 깊이(20층)·파라미터(0.27M)·학습 설정이 모두 같습니다. `out + self.shortcut(x)` 한 줄의 유무가 정확히 어떤 차이를 만드는지 측정합니다.

**2. 학습된 ResNet은 정말 잔차를 0에 가깝게 학습하는가?** ⭐
논문 §3.1의 핵심 가설은 *"F(x)가 0에 가깝기 때문에 학습이 쉽다"* 입니다. 두 모델의 BatchNorm 출력 표준편차를 layer별로 측정(논문 §4.2 Figure 7 재현)해 이 가설을 데이터로 확인합니다.

**결과 요약**:
- ResNet-20: **89.16%** vs Plain-20: **86.97%** → **+2.19%p** (Skip Connection의 효과)
- ResNet의 BN 출력 평균 std가 Plain보다 **10.7% 작음** → 잔차 가설 입증

---

## 📂 프로젝트 구조 (Project Structure)

```
├─ src/
│  └─ resnet_from_scratch.py            # 모델 + 학습 + 시각화 통합 스크립트
├─ results/
│  ├─ fig_01_training_curves.png        # 학습 곡선 (Plain-20 vs ResNet-20)
│  ├─ fig_02_accuracy_comparison.png    # 정확도 격차 + Skip Connection 효과 어노테이션
│  └─ fig_03_layer_response.png         # ⭐ Layer Response (논문 §4.2 Fig.7 재현)
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

---

## 🏗️ ResNet의 메커니즘과 구현 (Core Mechanism & Implementation)

논문의 6대 핵심 메커니즘을 *이론(Theory) → 구현(Implementation)* 한 자리에 정리합니다. 각 행이 한 메커니즘의 *핵심 개념* 과 *본 포트폴리오에서의 구현·검증 방식* 을 함께 보여줍니다.

### 📋 6대 메커니즘 요약

| # | 메커니즘 *(논문 §)* | 핵심 개념 (Theory) | 구현 · 검증 (Implementation) |
|:-:|:------------------|:------------------|:--------------------------|
| 1 | **Degradation Problem** *(§1)* | 깊은 plain net 의 **학습 오류조차도** 얕은 net 보다 높음 → *표현력의 한계가 아니라 최적화의 어려움*. 논문 인용: *"깊은 모델은 적어도 얕은 모델만큼은 해야 한다(identity layer만 추가하면 됨), 그런데 SGD가 그 해를 못 찾는다"* | 본 실험이 §2 잔차 학습으로 이 문제를 해결할 수 있음을 직접 검증 (fig_01 · 02) |
| 2 | **Residual Learning** *(§3.1, §3.2)* | 기존: `H(x)` 를 **직접** 학습 → ResNet: `H(x) = F(x) + x`, **잔차 `F(x) = H(x) − x` 만 학습**. 최적해가 identity 매핑에 가까울 때 `F(x) → 0` 으로 보내는 게 conv·BN 으로 identity 를 흉내내기보다 훨씬 쉬움 | `BasicBlock`: 3×3 Conv·BN 두 번 후 forward 끝에 `out + self.shortcut(x)` 한 줄로 입력 더함 (§3.2 Option B). `PlainBlock`: 그 한 줄만 빠진 블록. **두 블록의 유일한 차이는 정확히 그 한 줄** |
| 3 | **Skip Connection 의 Gradient 효과** *(§3.1)* | Plain: gradient = `W_N · ... · W_1` 누적 → 1 미만 가중치는 0 으로 수렴(**Vanishing**). ResNet: 미분하면 `∂L/∂x = ∂L/∂y · (∂F/∂x + 1)` — **"+1"** 항이 항상 살아있어 `∂F/∂x` 가 어떤 값이든 gradient 가 0이 되지 않음 | 152층까지 학습 가능한 *수학적 근거*. fig_01 학습 곡선의 일관된 격차가 이 효과의 정량 증거 |
| 4 | **CIFAR-10 6n+2 Architecture** *(§4.2)* | `Conv(3×3, 16) → Stage 1·2·3 (각 n block, {16, 32, 64} filter, stride=2 다운샘플) → GAP → FC(10)` 의 6n+2 층 구조. `n=3` 이면 ResNet-20 (아래 상세 표) | `CIFAR10Net(n=3, block=BasicBlock 또는 PlainBlock)` — 한 클래스로 두 모델 모두 표현. **block 추상화 하나로 두 모델이 동일한 학습 루프를 공유** |
| 5 | **Layer Response 가설** ⭐ *(§4.2 Fig.7)* | *"학습된 ResNet 의 layer 응답이 plain network 보다 작다 → 잔차 F(x) 가 실제로 0에 가까운 함수"*. 논문에서 분량은 짧지만 ResNet 핵심 가설을 뒷받침하는 **가장 결정적인 실증** | 학습된 두 모델의 모든 BatchNorm 출력에 PyTorch forward hook 을 걸어 layer별 std 측정 → fig_03 에서 정량 검증 |
| 6 | **Hyperparameters** *(§3.4, §4.2)* | SGD (momentum=0.9, weight_decay=1e-4), LR 0.1 → 분기점 ÷10, 4-pixel padding + random crop + horizontal flip, He init (`Var(W) = 2/n_in`, ReLU 에 맞춤) | 논문 설정 그대로. Epochs 만 빠른 데모용으로 30 (논문은 200, LR 분기점 [15, 22] = 논문 [100, 150] / 200 의 비율 유지) |

### 📐 6n+2 Architecture 상세 *(메커니즘 #4 의 구체)*

| Stage | 구성 | 출력 크기 |
|:------|:-----|:---------|
| Conv1 | 3×3 Conv, 16 filters | 32×32×16 |
| Stage 1 | n × Block (16→16) | 32×32×16 |
| Stage 2 | n × Block (16→32, stride=2) | 16×16×32 |
| Stage 3 | n × Block (32→64, stride=2) | 8×8×64 |
| Pool | Global Average Pooling | 64 |
| FC | Linear(64→10) | 10 |

> 더 깊은 ResNet-50/101/152 는 `BasicBlock` 대신 **Bottleneck Block** (1×1 → 3×3 → 1×1) 을 사용해 계산량을 1/9 로 줄이지만, 본 CIFAR-10 실험에서는 사용하지 않습니다.
>
> **Batch Normalization** (Ioffe & Szegedy, 2015) 과 **He Initialization** (He et al., 2015 ICCV) 은 ResNet 논문의 기여가 아니라 *§3.4 가 사용하는 외부 기술* 입니다. 두 기술 없이는 깊은 학습이 시작조차 안 되므로, 본 구현이 정확히 적용했음을 명시합니다.

---

## 📊 실험 결과 (Experimental Results)

### 1. 학습 곡선 — Plain-20 vs ResNet-20

![training curves](results/fig_01_training_curves.png)

같은 LR 스케줄·같은 데이터·같은 초기화에서 학습된 두 모델의 테스트 정확도. **학습 초반부터 ResNet-20(파랑)이 Plain-20(빨강)보다 대체로 위에** 위치하고, LR 분기점(epoch 15·22) 이후에는 격차가 안정적으로 유지됩니다. 두 모델이 *서로 다른 최적화 경로* 를 따라간다는 뜻이고, 최종 +2.19%p 격차는 학습이 끝날 때까지 메워지지 않습니다 — Skip Connection 한 줄이 만드는 차이가 *우연*이 아니라 *구조적 차이* 에서 비롯된다는 첫 번째 증거입니다.

LR 분기점(epoch 15, 22)에서 두 모델 모두 정확도가 점프하는 패턴이 보이는데, 이는 *"넓게 탐색하다가 점점 좁히는"* MultiStep LR 전략이 잘 작동하고 있음을 의미합니다.

### 2. 정확도 격차 — Skip Connection의 순수 효과

![accuracy comparison](results/fig_02_accuracy_comparison.png)

같은 0.27M 파라미터·같은 20층 깊이에서 **Skip Connection 한 줄**만 다른 두 모델의 최고 정확도. **+2.19%p의 격차** 는:

- ❌ 모델 크기 차이 — *둘 다 0.27M 파라미터*
- ❌ 깊이 차이 — *둘 다 20층*
- ❌ 학습 설정 차이 — *모두 동일*
- ✅ **순수하게 `out + shortcut(x)` 한 줄이 만드는 격차**

논문 §1의 주장 — *"Degradation은 표현력 부족이 아니라 최적화의 문제"* — 이 정확도 숫자 하나로 정확히 입증됩니다.

### 3. Layer Response — 잔차 가설을 데이터로 검증 ⭐

![layer response](results/fig_03_layer_response.png)

논문 §3.1의 핵심 가설 — *"잔차 F(x)는 일반적으로 0에 가깝다"* — 를 검증한 자리입니다. 학습된 두 모델의 모든 `BatchNorm2d` 모듈에 PyTorch forward hook을 걸어 출력의 표준편차를 8 batch 평균으로 측정한 뒤, 논문 Figure 7 우측 panel과 동일한 형식으로 **std 내림차순 정렬**해 두 곡선의 위치 관계를 비교했습니다.

**결과**: ResNet-20의 평균 응답(0.781)이 Plain-20(0.875)보다 **약 10.7% 작음**. 첫 두 layer 에서는 두 모델이 비슷하거나 ResNet 이 잠깐 위로 올라오는 구간이 있지만, **layer 2 이후부터 학습 마지막 layer 까지 ResNet 이 Plain 보다 안정적으로 작은 값을 유지**합니다. 이는 Skip Connection이 *gradient를 살리는 트릭*에 그치지 않고, **모델이 잔차 F(x)를 0에 가까운 함수로 학습하도록 유도하는 구조적 편향(inductive bias)** 임을 보여줍니다.

논문 §3.1의 가설이 — 정확도 숫자가 아닌 — **학습된 모델 가중치의 행동**으로 입증되는 자리입니다. 이게 본 포트폴리오의 가장 결정적인 발견입니다.

---

## ✨ 분석 및 발견 (Key Findings & Analysis)

### 1. Skip Connection 의 효과는 표현력이 아니라 최적화 문제였다

처음에는 같은 깊이·같은 파라미터에서 한 줄 차이로 +2.19%p 가 나는 게 잘 와닿지 않았습니다. 표현력(capacity) 으로만 따지면 Plain 도 identity 매핑 정도는 충분히 학습할 수 있어야 하니까요. 어떤 layer 가 입력을 그대로 통과시키도록 가중치를 맞추면 되는 단순한 일인데, fig_02 결과를 보면 같은 조건의 Plain 이 그걸 못 하고 있다는 게 분명히 보였습니다. 결국 차이는 *"Plain 이 표현할 수 있느냐"* 가 아니라 *"SGD 가 그 해를 찾을 수 있느냐"* 의 문제였고, 논문 §1 의 degradation problem 본질이 통제 비교 한 번으로 깔끔하게 분리됐습니다.

### 2. 잔차는 정말 0 에 가까웠다 ⭐

fig_03 을 그릴 때 결과가 어떻게 나올지 솔직히 자신이 없었습니다. ResNet 의 BN 응답이 정말로 Plain 보다 작을지, 비슷할지, 오히려 클지는 측정 전까지 가설로만 알고 있었거든요. 막상 측정해보니 평균 std 가 ResNet 0.781 vs Plain 0.875 로 약 10.7% 작았고, 첫 두 layer 를 제외하면 마지막 layer 까지 격차가 안정적으로 유지됐습니다. §3.1 의 *"잔차 F(x) 가 0 에 가까울수록 학습이 쉽다"* 가설이 학습된 가중치에 실제로 남아 있다는 의미였고, Skip Connection 이 단순한 gradient 우회로가 아니라 모델을 identity 근처에 머물게 하는 inductive bias 라는 점이 그제서야 와닿았습니다.

### 3. 이론과 실증, 두 효과가 결합되어야 깊은 망 학습이 동작한다

논문을 처음 읽을 때는 §3.1 의 두 주장이 서로 어떻게 연결되는지 헷갈렸습니다. 하나는 *"gradient 가 `(∂F/∂x + 1)` 의 +1 항 덕에 vanishing 되지 않는다"* 는 이론적 보장이고, 다른 하나는 *"학습된 모델의 잔차가 실제로 0 에 가깝다"* 는 실증인데, 둘이 별개의 이야기처럼 들렸습니다. 두 실험(fig_01·02 의 학습 가능성 + fig_03 의 layer response) 을 직접 해보고 나서야, 이 두 효과가 *결합* 되어야 152층까지 학습이 현실에서 동작한다는 게 정리됐습니다. 이론만 있고 실제 모델이 복잡한 함수를 학습하려 들면 안 되고, 실증만 있고 gradient 가 죽어버리면 학습 자체가 시작도 안 되니까요.

### 4. MultiStep LR 의 효과가 학습 곡선의 모양으로 찍힌다

fig_01 에서 epoch 15 와 22 에 두 모델이 동시에 점프하는 모습이 가장 눈에 띄었습니다. 큰 LR 로 충분히 탐색한 뒤 두 번에 걸쳐 LR 을 ÷10 씩 줄이는 MultiStep 스케줄의 효과가, 수식이나 설명이 아니라 곡선의 두 차례 점프로 그대로 보였습니다. 논문이 200 epoch 동안 굳이 이 두 분기점만 고집한 이유가 그림 한 장으로 이해되는 순간이었습니다.

---

## 💡 회고록 (Retrospective)

라이브러리 한 줄(`torchvision.models.resnet50`)로 끝낼 수 있는 ResNet 을 PyTorch 기본 연산만으로 처음부터 짜봤습니다. 개념 자체는 단순합니다. Conv 두 개에 입력을 한 번 더해주는 게 전부니까요. 그런데 막상 직접 짜고 학습 곡선을 그려보니 그 한 줄(`out + self.shortcut(x)`) 이 만드는 차이가 생각보다 컸습니다.

처음 만든 버전은 지금보다 훨씬 컸습니다. ResNet-20/32/56 세 깊이를 다 학습해서 논문 Table 6 를 재현하고, Shortcut Option A/B/C 세 가지를 전부 구현하고, BN 과 He init 의 효과까지 별도로 시각화했었습니다. 그런데 다 만들어놓고 보니 *"이 포트폴리오의 핵심 메시지가 뭔가?"* 라는 질문에 답하기가 애매했습니다. 깊이 비교(20→32→56)는 ResNet 튜토리얼에서 흔히 다루는 내용이었고, BN/He init 은 ResNet 의 기여가 아니라 외부 기술이었는데 마치 ResNet 의 일부처럼 다뤄지고 있었거든요. 정작 보여주고 싶었던 건 *"같은 조건에서 Skip Connection 한 줄이 만드는 차이"* 와 *"잔차 F(x) 가 정말 0 에 가까운가"* 두 가지였는데, 부가 콘텐츠가 그 메시지를 가리고 있었습니다. 결국 코드를 1,000 줄에서 280 줄로, 시각화를 10 개에서 3 개로 줄였습니다. 줄일 때는 만들어 놓은 게 아까웠는데, 정리하고 나니 보여주려는 게 명확해졌습니다.

흥미로웠던 건 Plain-20 과 ResNet-20 을 같은 조건으로 돌려본 부분이었습니다. *"Skip Connection 한 줄로 성능이 올라간다"* 는 말은 어디서나 들을 수 있지만, 두 모델을 직접 같은 seed 로 학습시켜놓고 epoch 1 부터 30 까지 격차가 좁혀지지 않는 걸 보고 나서야 *"학습이 덜 끝나서 그런 게 아니구나"* 가 와닿았습니다. 두 모델이 처음부터 다른 경로로 최적화되고 있다는 뜻이었고, 결국 Skip Connection 한 줄이 loss landscape 자체를 다르게 만든다는 의미였습니다.

Layer Response 분석(fig_03) 을 마지막에 추가하면서 논문을 그제서야 제대로 읽었다는 느낌이 들었습니다. 그 전까지는 *"ResNet 이 잘 되는 이유는 gradient vanishing 을 해결하기 때문"* 정도로만 알고 있었는데, 학습된 두 모델의 BN 출력에 forward hook 을 걸어 std 를 측정해보니 ResNet 의 응답이 Plain 보다 평균 10.7% 작았습니다. §3.1 의 *"잔차가 0 에 가깝다"* 라는 가설이 학습된 가중치 안에 실제로 남아 있다는 의미였습니다. 논문에서는 §4.2 Figure 7 한 페이지로 짧게 다뤄지지만, 직접 그려보고 나서야 Skip Connection 이 단순히 gradient 우회로가 아니라 모델을 identity 근처로 끌어당기는 inductive bias 라는 의미가 와닿았습니다. *"잘 된다"* 에서 끝나지 않고 *"왜 잘 되는지"* 까지 데이터로 들여다본 게 처음이라, 이번 프로젝트에서 가장 기억에 남습니다.

다음에는 이 코드를 베이스로 Pre-activation ResNet (*Identity Mappings in Deep Residual Networks*) 을 비교해보고 싶습니다. Skip Connection 의 위치(post-activation vs pre-activation) 가 fig_03 의 Layer Response 패턴을 어떻게 바꾸는지가 궁금합니다.

---

## 🔗 참고 자료 (References)

- **He, K., et al.** "Deep Residual Learning for Image Recognition." *CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- He, K., et al. "Identity Mappings in Deep Residual Networks." *ECCV*, 2016. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027) — *Pre-activation ResNet*
- Ioffe, S., Szegedy, C. "Batch Normalization." *ICML*, 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) — *BN 원논문 (ResNet이 사용)*
- He, K., et al. "Delving Deep into Rectifiers." *ICCV*, 2015. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) — *He init 원논문 (ResNet이 사용)*
- PyTorch Docs — `nn.Conv2d`, `nn.BatchNorm2d`, `nn.init.kaiming_normal_`, `optim.SGD`, `MultiStepLR`
