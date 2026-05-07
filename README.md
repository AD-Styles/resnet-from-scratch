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
| 1 | **Degradation Problem** *(§1)* | 깊은 plain net 의 학습 오류조차 얕은 net 보다 높음 → 표현력 한계가 아니라 **최적화의 어려움** | 본 실험이 §2 잔차 학습으로 이 문제를 해결할 수 있음을 검증 (fig_01·02) |
| 2 | **Residual Learning** *(§3.1, §3.2)* | `H(x)` 를 직접 학습하는 대신 잔차 `F(x) = H(x) − x` 만 학습. 최적해가 identity 에 가까우면 `F(x) → 0` 으로 보내는 편이 훨씬 쉬움 | `BasicBlock` forward 끝의 `out + self.shortcut(x)` 한 줄(§3.2 Option B). `PlainBlock` 은 그 한 줄만 빠짐 — **유일한 차이** |
| 3 | **Skip Connection 의 Gradient 효과** *(§3.1)* | 미분하면 `∂L/∂x = ∂L/∂y · (∂F/∂x + 1)` — **"+1" 항이 항상 살아있어** gradient 가 0 이 되지 않음 (Plain 은 가중치 곱이 누적되며 vanishing) | 152층까지 학습 가능한 수학적 근거. fig_01 의 일관된 격차가 정량 증거 |
| 4 | **CIFAR-10 6n+2 Architecture** *(§4.2)* | Conv → Stage 1·2·3 (각 `n` block, {16, 32, 64} filter) → GAP → FC. **`n=3` 이면 ResNet-20** *(상세는 아래 표)* | `CIFAR10Net(n=3, block=...)` — 한 클래스로 두 모델 표현, 동일 학습 루프 공유 |
| 5 | **Layer Response 가설** ⭐ *(§4.2 Fig.7)* | *"학습된 ResNet 의 layer 응답이 Plain 보다 작다 → 잔차 F(x) 가 0 에 가까운 함수"*. 논문 분량은 짧지만 ResNet 핵심 가설의 **가장 결정적인 실증** | 두 모델의 모든 BatchNorm 출력에 forward hook → layer 별 std 측정 (fig_03) |
| 6 | **Hyperparameters** *(§3.4, §4.2)* | SGD (momentum 0.9, weight_decay 1e-4), LR 0.1 → 분기점에서 ÷10, 4-pixel padding + random crop + horizontal flip, He init | 논문 설정 그대로. Epochs 만 데모용 30 (논문 200, LR 분기점 비율 유지) |

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

| 발견 | 의미 |
|:-----|:-----|
| **Skip Connection의 효과는 표현력이 아니라 최적화** | 같은 깊이·같은 파라미터에서 ResNet이 Plain보다 +2.19%p 좋음. 격차의 원인이 *모델이 표현할 수 있는 함수의 종류*가 아니라 *SGD가 좋은 해를 찾을 수 있는가*의 문제라는 게 통제 비교로 분리됨 |
| **잔차는 정말 0에 가까웠다** ⭐ | ResNet의 BN 출력 std가 Plain보다 평균 10.7% 작음. 논문 §3.1 가설이 학습된 가중치의 행동으로 입증됨. *Skip Connection은 단지 gradient 우회로가 아니라 모델을 identity 근처에 머물게 하는 inductive bias* |
| **두 효과가 결합되어 깊은 망 학습이 가능** | (1) gradient flow에서 `(∂F/∂x + 1)`의 "+1" 항이 vanishing 방지 — *이론* (2) 실제로 모델이 F(x)를 0에 가깝게 학습 — *실증*. 두 효과가 결합되어 152층 학습이 현실에서 동작 |
| **LR MultiStep의 효과가 곡선에서 보임** | epoch 15, 22에서 LR을 10배씩 낮출 때마다 두 모델 모두 정확도가 점프. *"넓게 탐색 → 두 번 좁히기"* 전략의 효과가 학습 곡선에 그대로 찍힘 |

---

## 💡 회고록 (Retrospective)

라이브러리 한 줄(`torchvision.models.resnet50`)로 끝낼 수 있는 ResNet 을 PyTorch 기본 연산만으로 처음부터 짜봤습니다. 개념 자체는 단순합니다. Conv 두 개에 입력을 한 번 더해주는 게 전부입니다. 그런데 막상 직접 짜고 학습 곡선을 그려보니 그 한 줄(`out + self.shortcut(x)`) 이 만드는 차이가 생각보다 컸습니다.

처음 만든 버전은 지금보다 훨씬 컸습니다. ResNet-20/32/56 세 깊이를 다 학습해서 논문 Table 6 를 재현하고, Shortcut Option A/B/C 세 가지를 전부 구현하고, BN 과 He init 의 효과까지 별도로 시각화했었습니다. 그런데 다 만들어놓고 보니 *"이 포트폴리오의 핵심 메시지가 뭔가?"* 라는 질문에 답하기가 애매했습니다. 깊이 비교(20→32→56)는 ResNet 튜토리얼에서 흔히 다루는 내용이었고, BN/He init 은 ResNet 의 기여가 아니라 외부 기술이었는데 마치 ResNet 의 일부처럼 다뤄지고 있었습니다. 정작 보여주고 싶었던 건 *"같은 조건에서 Skip Connection 한 줄이 만드는 차이"* 와 *"잔차 F(x) 가 정말 0 에 가까운가"* 두 가지였는데, 부가 콘텐츠가 그 메시지를 가리고 있었습니다. 결국 코드를 1,000 줄에서 280 줄로, 시각화를 10 개에서 3 개로 줄였습니다. 줄일 때는 만들어 놓은 게 아까웠는데, 정리하고 나니 보여주려는 게 명확해졌습니다.

흥미로웠던 건 Plain-20 과 ResNet-20 을 같은 조건으로 돌려본 부분이었습니다. *"Skip Connection 한 줄로 성능이 올라간다"* 는 말은 어디서나 들을 수 있지만, 두 모델을 직접 같은 seed 로 학습시켜놓고 epoch 1 부터 30 까지 격차가 좁혀지지 않는 걸 보고 나서야 *"학습이 덜 끝나서 그런 게 아니구나"* 가 와닿았습니다. 두 모델이 처음부터 다른 경로로 최적화되고 있다는 뜻이었고, 결국 Skip Connection 한 줄이 loss landscape 자체를 다르게 만든다는 의미였습니다.

fig_03 작업이 가장 인상 깊었습니다. 사실 그 전까지는 *"ResNet 이 잘 되는 이유는 gradient vanishing 을 해결하기 때문"* 정도로만 알고 있었고, 그 이상은 굳이 파고든 적이 없었습니다. 그런데 두 모델을 다 학습시켜놓고 BN 출력에 forward hook 을 걸어 std 를 직접 찍어보니, ResNet 의 평균 응답이 Plain 보다 10.7% 작게 나왔습니다. 솔직히 처음에는 이 숫자가 정확히 뭘 뜻하는지 바로 안 와닿았습니다. §3.1 을 다시 읽어보고 나서야 *"잔차 F(x) 가 0 에 가깝다"* 는 가설을 지금 데이터로 직접 확인하고 있다는 게 정리됐습니다. 논문에서는 §4.2 Figure 7 한 페이지로 짧게 지나가는 부분인데, 막상 직접 그려놓고 보니 그 한 장이 ResNet 가설의 가장 결정적인 증거였습니다. Skip Connection 을 *gradient 우회로* 정도로만 보던 시각이, *모델을 identity 근처로 끌어당기는 inductive bias* 라는 좀 더 구체적인 의미로 바뀐 순간이었습니다. 가중치가 실제로 어떻게 행동하는지까지 데이터로 들여다본 건 이번이 처음이라, 프로젝트에서 가장 기억에 남는 작업이 됐습니다.

다음에는 이 코드를 베이스로 Pre-activation ResNet (*Identity Mappings in Deep Residual Networks*) 을 비교해보고 싶습니다. Skip Connection 의 위치(post-activation vs pre-activation) 가 fig_03 의 Layer Response 패턴을 어떻게 바꾸는지가 궁금합니다.

---

## 🔗 참고 자료 (References)

- **He, K., et al.** "Deep Residual Learning for Image Recognition." *CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- He, K., et al. "Identity Mappings in Deep Residual Networks." *ECCV*, 2016. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027) — *Pre-activation ResNet*
- Ioffe, S., Szegedy, C. "Batch Normalization." *ICML*, 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) — *BN 원논문 (ResNet이 사용)*
- He, K., et al. "Delving Deep into Rectifiers." *ICCV*, 2015. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) — *He init 원논문 (ResNet이 사용)*
- PyTorch Docs — `nn.Conv2d`, `nn.BatchNorm2d`, `nn.init.kaiming_normal_`, `optim.SGD`, `MultiStepLR`
