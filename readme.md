# DCSPG 학습 프레임워크

이 문서는 [`train_dcspg_framework.py`](train_dcspg_framework.py)를 진입점으로 사용하는
Data-Conditioned Symbolic Proxy Generator(DCSPG)의 데이터 구성, 모델, 학습, validation,
test 과정을 설명한다. 설명과 기본값은 현재 코드 기준이다.

## 1. 목표

DCSPG는 unseen time-series dataset에서 소수 sample의 통계만 관측하고, 해당 dataset에서
forecasting model의 성능 순위를 잘 대리할 symbolic proxy formula를 생성한다.

모델의 입력과 출력은 다음과 같다.

```text
K=16개 time-series sample
        │
        ├─ sample마다 계산된 22개 catch22 feature
        │
        └─ feature별 mean/std 집계: [22, 2]
                         │
                         ▼
              Catch22 Transformer Encoder
                         │
                         ▼
              RPN Transformer Decoder
                         │
                         ▼
       예: Jacob_fro Snip Div GSynFlow Add <EOS>
```

생성된 수식은 benchmark의 base proxy 값에 적용된다. 수식이 산출한 architecture 순위와
forecasting 성능인 `-MSE` 순위 사이의 Spearman correlation이 최종 품질 지표다. 따라서
이 프레임워크는 forecasting MSE를 직접 예측하거나 forecasting model을 직접 학습하지
않는다.

## 2. 주요 코드

| 파일 | 역할 |
|---|---|
| `train_dcspg_framework.py` | CLI, 설정 검증, 39/8/6 split 구성, 학습 실행 |
| `DCSPG/data.py` | catch22 NPZ 로딩, support episode 및 cluster-balanced batch sampling |
| `DCSPG/targets.py` | dataset별 weighted teacher formula 로딩 및 sampling |
| `DCSPG/model.py` | catch22 encoder와 autoregressive symbolic decoder |
| `DCSPG/grammar.py` | 생성 단계별 RPN 문법 mask |
| `DCSPG/trainer.py` | multi-teacher weighted CE와 optimizer step |
| `DCSPG/split_training.py` | 전체 train/validation/test loop, checkpoint 및 결과 저장 |
| `DCSPG/evaluate.py` | RPN 실행, proxy score 계산, Spearman 평가 |
| `DCSPG/dataset_partition.py` | dataset alias 처리 및 39/8/6 partition |
| `DCSPG/vocabulary.py` | symbolic token vocabulary와 ID 변환 |

## 3. 입력 데이터

### 3.1 Time-series feature

기본 경로는 `DCSPG/TS_dataset/*.npz`다. 각 파일의 `features` 배열은 `[N, 22]`
형태이며, `N`개 time-series sample 각각에 대한 catch22 feature를 담는다.

하나의 episode를 만들 때 dataset에서 기본 `K=16`개 sample을 비복원 추출한다. 이후
22개 feature 각각에 대해 population mean과 standard deviation(`ddof=0`)을 계산하여
다음 입력을 만든다.

```text
episode input shape = [22, 2]
last dimension       = [mean, std]
meta-batch shape     = [B, 22, 2]
```

NaN/Inf 집계 결과는 0으로 대체된다. dataset의 sample 수가 `K`보다 작으면 fixed-split
sampler는 오류를 발생시킨다.

### 3.2 Ground-truth teacher formula

기본 경로는 `DCSPG/GroundTruth/*.npz`다. 각 dataset 파일은 최소한 다음을 가진다.

- `rpn_tokens`: symbolic proxy의 RPN token sequence
- `weight`: teacher formula의 품질 가중치

현재 GroundTruth 생성 정책에서는 formula가 validation fitness 내림차순으로 정렬되며,
가중치는 최고 formula의 `1.0`부터 마지막 formula의 `0.1`까지 percentile rank에 따라
감소한다.

기본 `teachers_per_episode=16`이므로 한 episode에 해당 dataset의 teacher formula 16개를
균등 비복원 추출한다. 이 경우 `--target-sampling-strategy`의 `cycle/random` 차이는
적용되지 않는다. 해당 전략은 teacher를 한 개만 뽑는 호환 모드에서만 의미가 있다.

### 3.3 Proxy benchmark

Validation과 test에는 각 forecasting candidate의 base proxy 값과 MSE가 저장된 CSV가
필요하다.

- 47개 Monash/TIME 계열: `proxy_scores/monash_time`
- 6개 benchmark dataset: `DCSPG/Benchmark`

기본 평가 split은 `proxy_test`이고, `status == "success"`인 행만 사용한다. 목표 벡터는
MSE에 음수를 취한 `-MSE`다. 따라서 높은 proxy score가 좋은 forecasting 성능과 같은
방향을 갖도록 평가된다.

## 4. Dataset partition과 meta-batch

전체 53개 dataset은 정확히 다음과 같이 분리된다.

- Train: validation/test에 포함되지 않은 39개
- Validation: 8개
- Test: 6개

기본 validation dataset은 다음과 같다.

```text
Coastal_T_S__H
sunspot_dataset_without_missing_values
Australia_Solar__H
Water_Quality_Darwin__15T
current_velocity__20T
wind_4_seconds_dataset
SG_Carpark__15T
Port_Activity__D
```

기본 test dataset은 `ECL`, `Exchange`, `Illness`, `Traffic`, `Weather`, `ETTh1`이다.
Benchmark alias는 TS feature 이름과 GroundTruth/proxy-score 이름으로 자동 변환된다.

Train dataset은 catch22 centroid 기반 4개 cluster로 나뉜다. 매 iteration마다 각
cluster에서 기본 8 episode를 뽑으므로 batch 크기는 다음과 같이 고정된다.

```text
4 clusters × 8 episodes/cluster = batch size 32
```

Cluster 안에서는 dataset을 균등 복원 추출한다. 선택된 dataset 안에서는 support sample
16개를 비복원 추출한다. 이 구성은 cluster 크기가 다르더라도 각 cluster가 batch에 같은
비중으로 기여하게 한다. `--batch-size`는 항상 `4 × --episodes-per-cluster`여야 한다.

## 5. 모델 설계

### 5.1 기본 크기

| 항목 | 기본값 |
|---|---:|
| `d_model` | 64 |
| attention heads | 2 |
| encoder layers | 2 |
| decoder layers | 2 |
| feed-forward dimension | 128 |
| dropout | 0.1 |
| maximum formula length | 12 |
| maximum RPN stack depth | 4 |
| maximum consecutive unary operators | 2 |

CLI 기본값은 `DCSPGConfig` dataclass 자체의 일부 기본값보다 우선한다.

### 5.2 Value/feature embedding

입력 `stats[B, 22, 2]`에서 음수 standard deviation을 0으로 clamp한 후 각 feature의
값 표현을 다음처럼 만든다.

```text
value_j   = Linear([mean_j, log1p(std_j)])
feature_j = Embedding(feature_id_j)
token_j   = Dropout(LayerNorm(value_j + feature_j))
```

`feature_id` embedding은 서로 다른 catch22 feature의 의미를 구분한다. Encoder용 별도
position embedding은 없지만, feature identity embedding이 각 입력 열을 식별한다.

### 5.3 Catch22 encoder

학습 가능한 `[CLS]` token을 22개 feature token 앞에 붙여 총 23개 token을 만든다.
Transformer encoder는 모든 feature 사이의 관계를 self-attention으로 결합한다.

- `memory[B, 23, 64]`: decoder cross-attention에 전달되는 전체 encoder 출력
- `context[B, 64]`: encoder 출력의 `[CLS]` 위치로, dataset-level 조건 벡터

### 5.4 Symbolic autoregressive decoder

Decoder는 token embedding과 학습 가능한 position embedding을 사용한다. 첫 decoder
위치에는 별도 `<BOS>` token 대신 encoder의 `context`가 들어간다. 그 뒤에는 이미 알려진
또는 생성된 RPN prefix가 들어간다.

```text
decoder input = [context@pos0, rpn_token_1@pos1, ..., rpn_token_t@post]
```

Causal self-attention은 미래 token 참조를 막고, cross-attention은 encoder의 23개
`memory` token 전체를 참조한다. 마지막 linear projection이 vocabulary logits를 만든다.

학습에서는 teacher forcing을 사용한다. 예를 들어 target이 `[a, b, Mul, EOS]`라면
decoder prefix는 `[a, b, Mul]`이고, context 위치부터 차례로 `a`, `b`, `Mul`, `EOS`를
예측한다. 현재 `make_decoder_inputs()`의 `bos_id` 인자는 API 호환용이며 실제 token
sequence 앞에 BOS를 삽입하지 않는다.

### 5.5 Vocabulary와 RPN grammar

기본 symbolic token은 다음 세 종류다.

- Operand: `Fisher`, `GFLOPs`, `GSynFlow`, `GraSP`, `Grad_Norm`, `Jacob_fro`,
  `Jacov`, `L2-Norm`, `MParams`, `Snip`, `ZiCo`, `plain`
- Unary operator: `Square`, `Sqrt`, `Log`, `Negative`
- Binary operator: `Add`, `Sub`, `Mul`, `Div`

여기에 `<pad>`, `<bos>`, `<eos>`, `<unk>`가 추가된다. GroundTruth에서 발견된 추가
token도 vocabulary에 병합된다.

RPN grammar는 train과 inference 모두에서 invalid token의 logit을 사실상 `-inf`로
mask한다.

- Operand: stack depth를 1 증가
- Unary operator: stack에 값이 하나 이상 있을 때만 허용, depth 유지
- Binary operator: stack에 값이 두 개 이상 있을 때만 허용, depth를 1 감소
- EOS: stack depth가 정확히 1일 때만 허용
- 종료 후: PAD만 허용
- Stack depth: 기본 최대 4
- 연속 unary operator: 기본 최대 2
- 남은 길이 안에 완성할 수 없는 선택도 금지

연속 unary 횟수는 operand 또는 binary operator가 나오면 0으로 초기화된다. 두 unary
operator가 연속된 상태에서는 다음 unary token 전체를 mask하므로 길이 3 이상의 chain은
생성할 수 없다. 같은 `RPNGrammar`가 training teacher forcing, validation/test greedy
generation, stochastic sampling, beam search에 공통 적용된다.

문법은 syntactic validity를 보장하지만 division overflow, non-finite score, constant
ranking처럼 실행 시 발생하는 numerical invalidity까지 보장하지는 않는다.

## 6. 학습 방식

### 6.1 한 iteration의 흐름

1. 네 cluster에서 각각 8 episode를 뽑아 32개 episode batch를 만든다.
2. 각 episode에서 16개 sample의 catch22 mean/std를 계산한다.
3. 해당 dataset GroundTruth에서 teacher formula 16개를 비복원 추출한다.
4. Encoder는 episode를 한 번만 encode한다.
5. 동일한 encoder context/memory를 16개 teacher에 복제한다.
6. Decoder가 teacher forcing으로 각 formula token과 EOS를 예측한다.
7. 문법 mask 후 teacher별 token-mean cross entropy를 계산한다.
8. Teacher weight를 episode 내부 합이 1이 되도록 정규화한다.
9. Episode별 weighted teacher loss를 구하고, 32개 episode 평균을 최종 loss로 사용한다.
10. Backpropagation, gradient clipping, AdamW update를 수행한다.

Teacher `m`의 token-mean CE를 `L_m`, NPZ weight를 `w_m`이라 하면 episode loss는 다음과
같다.

```text
normalized_w_m = w_m / (sum_j w_j + 1e-8)
episode_loss   = sum_m normalized_w_m * L_m
batch_loss     = mean_episode(episode_loss)
```

PAD 위치는 CE에서 제외된다. Optimizer 기본값은 AdamW, learning rate `2e-4`, weight
decay `1e-4`이며 gradient norm은 기본 `1.0`으로 clip한다. 별도 learning-rate
scheduler나 warmup은 현재 사용하지 않는다.

### 6.2 Epoch와 logging

현재 CLI 기본값은 epoch당 20 iteration, 최대 500 epoch다. 매 20 global step마다 최근
loss와 dataset 선택 횟수를 출력한다. 각 epoch 종료 후 validation을 한 번 수행한다.

## 7. Validation

Validation support는 매 epoch 다시 sampling하지 않는다. Dataset 이름, stage 이름,
episode 번호를 SHA-256으로 hashing하여 training seed와 무관한 고정 support indices를
만든다.

기본 validation 구성은 다음과 같다.

```text
8 datasets × 5 fixed-support episodes = 40 generated formulas per epoch
generation = greedy decoding
evaluation split = proxy_test
```

각 생성 수식은 benchmark의 base proxy column을 입력으로 받아 candidate별 proxy score를
계산한다. 이어서 다음 값을 구한다.

```text
formula score = Spearman(generated_proxy_scores, -MSE)
```

각 dataset의 5개 episode Spearman을 먼저 평균하고, 다시 8개 dataset 평균을 내어
validation criterion을 만든다. 이는 dataset별 benchmark row 수 차이가 criterion의
가중치에 영향을 주지 않는 macro average다.

동일한 fixed-support episode에 대해 dataset-wise weighted validation CE도 계산한다.
각 episode와 해당 dataset의 모든 GroundTruth teacher formula 조합을 teacher forcing으로
평가한다. Formula별 CE는 PAD를 제외한 valid token 평균이며, dataset의 기존 rank-based
teacher weight를 다음처럼 정규화한다.

```text
episode CE = sum_teacher(weight × length-normalized CE) / sum_teacher(weight)
dataset CE = 5개 fixed episode CE의 평균
macro CE   = 8개 dataset CE의 동일 가중 평균
```

Teacher formula는 `--validation-ce-teacher-batch-size` 단위로 나누어 계산하지만, 이는
GPU memory 사용량만 바꾸며 모든 teacher를 포함한 최종 weighted mean 값에는 영향을
주지 않는다. 이 CE는 생성 수식과 teacher 수식을 비교하는 값이 아니라 teacher formula
자체를 decoder target으로 넣은 teacher-forcing CE다. Spearman checkpoint 선택 기준은
그대로 유지되며 CE는 별도의 진단 지표로 기록된다.

수식 parsing/evaluation 실패, non-finite 값, 유효하지 않은 Spearman은 기본 `-1.0`의
penalty로 dataset 평균에 포함된다. `--invalid-spearman-penalty`로 변경할 수 있다.

선택한 validation criterion이 이전 최적값보다 엄격히 개선될 때만 improvement로
인정한다. Spearman은 증가, CE는 감소가 개선이다. 기본 patience 10 epoch 동안 개선이
없으면 early stopping한다.

Early-stopping 기준은 선택할 수 있다.

```bash
# 기본값: macro Spearman을 최대화
--early-stopping-criterion spearman_corr

# Dataset-wise weighted macro validation CE를 최소화
--early-stopping-criterion celoss
```

`--validation-criterion`은 같은 option의 alias다. `celoss`를 선택해도 기존 greedy
formula 생성, dataset별/macro Spearman 계산, CSV 기록, 그래프 및 터미널 출력은 그대로
수행된다. Early stopping, `best_checkpoint.pth`를 포함한 top-checkpoint 순위, checkpoint
averaging 대상만 macro CE가 낮은 epoch를 기준으로 결정된다.

## 8. Checkpoint 선택과 averaging

매 epoch validation 결과를 기준으로 상위 checkpoint를 유지한다. 기본 설정에서는
`--averaged-checkpoint-count=3`이므로 상위 3개를 최종 보관하고 동일 가중치로 parameter
average하여 `averaged_checkpoint.pth`를 만든다.

```text
best_checkpoint.pth
second_best_checkpoint.pth
third_best_checkpoint.pth
averaged_checkpoint.pth = 위 3개 state_dict의 uniform average
```

Floating/complex tensor는 정밀도가 높은 dtype에서 평균한 후 원래 dtype으로 되돌린다.
Non-floating tensor는 checkpoint 사이에 값이 동일해야 한다.

별도로 최근 epoch checkpoint는 `checkpoints/epoch_XXXX.pth`에 저장되며
`--checkpoint-keep=5`에 따라 최근 5개만 남긴다. 즉, `checkpoint-keep`은 validation
상위 checkpoint 수가 아니라 rolling epoch checkpoint 수다. `last.pt`는 학습 종료 시점
모델이다.

## 9. Test

Test는 early stopping과 checkpoint ranking에 관여하지 않으며 학습 종료 후 한 번만
실행된다. Validation과 마찬가지로 training seed와 무관한 fixed support를 사용한다.

```text
6 datasets × 10 fixed-support repeats = checkpoint당 60개 formula
generation = greedy decoding
evaluation split = proxy_test
```

동일한 `test_batch`를 다음 두 모델에 각각 적용하므로 support 조건이 완전히 같다.

1. Validation 최고 단일 checkpoint
2. Validation 상위 checkpoint의 uniform weight average

각 dataset은 10개 formula의 penalized Spearman 평균으로 요약한다. 마지막 overall 값은
6개 dataset 평균의 macro average다. Test 결과는 모델 선택에 다시 사용되지 않는다.

Test 결과 CSV의 각 generated formula에는 `Avg. CE`도 기록된다. Dataset마다 생성된
10개 formula를 동일 dataset의 서로 다른 fixed support episode 10개에 각각
teacher-forcing target으로 적용하여 총 `10 × 10 = 100`개 sequence CE를 병렬 계산한다.
PAD를 제외한 length-normalized token CE를 사용하며, 각 formula에 대해 support episode
10개의 CE를 평균한 값이 해당 formula 행의 `Avg. CE`다.

```text
Avg. CE(formula_m) = mean over 10 fixed support episodes of CE(episode_e, formula_m)
```

## 10. 실행 방법

기본 실행:

```bash
conda run -n tslib_nightly python train_dcspg_framework.py --gpu-id 0
```

실시간 출력과 함께 직접 실행하려면:

```bash
conda activate tslib_nightly
python train_dcspg_framework.py --gpu-id 0
```

고유한 run 이름 지정:

```bash
conda run -n tslib_nightly python train_dcspg_framework.py \
  --gpu-id 0 \
  --run-name dcspg_seed2026 \
  --seed 2026
```

Validation/test dataset 변경:

```bash
conda run -n tslib_nightly python train_dcspg_framework.py \
  --validation-datasets "dataset_a,dataset_b,dataset_c,dataset_d,dataset_e,dataset_f,dataset_g,dataset_h" \
  --test-datasets "ECL,Exchange,Illness,Traffic,Weather,ETTh1" \
  --gpu-id 0
```

Validation은 정확히 8개, test는 정확히 6개여야 하며 서로 겹칠 수 없다. 나머지 39개
train dataset은 모두 cluster CSV에 할당되어 있어야 한다.

전체 option 확인:

```bash
conda run -n tslib_nightly python train_dcspg_framework.py --help
```

## 11. 주요 CLI 기본값

| Option | 기본값 | 의미 |
|---|---:|---|
| `--batch-size` | 32 | Meta-batch episode 수 |
| `--episodes-per-cluster` | 8 | Cluster별 episode 수 |
| `--k-samples` | 16 | Episode support sample 수 |
| `--teachers-per-episode` | 16 | Episode별 teacher formula 수 |
| `--d-model` | 64 | Transformer hidden dimension |
| `--n-heads` | 2 | Multi-head attention head 수 |
| `--encoder-layers` | 2 | Encoder layer 수 |
| `--decoder-layers` | 2 | Decoder layer 수 |
| `--dim-feedforward` | 128 | Transformer FFN dimension |
| `--max-formula-len` | 12 | EOS를 포함한 최대 생성 길이 |
| `--max-stack-depth` | 4 | 최대 RPN evaluation stack depth |
| `--max-unary-chain` | 2 | 허용할 최대 연속 unary operator 수 |
| `--learning-rate` | 2e-4 | AdamW learning rate |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--grad-clip` | 1.0 | Gradient norm clipping threshold |
| `--iterations-per-epoch` | 20 | Epoch당 optimizer step 수 |
| `--max-epochs` | 500 | 최대 epoch 수 |
| `--patience` | 10 | Early stopping patience |
| `--early-stopping-criterion` | spearman_corr | `spearman_corr` 최대화 또는 `celoss` 최소화 |
| `--validation-episodes-per-dataset` | 5 | Validation dataset별 fixed episode 수 |
| `--validation-ce-teacher-batch-size` | 128 | Validation 전체 teacher CE 계산 chunk 크기 |
| `--test-repeats` | 10 | Test dataset별 fixed episode 수 |
| `--invalid-spearman-penalty` | -1.0 | Invalid formula의 평균용 score |
| `--checkpoint-keep` | 5 | 최근 epoch checkpoint 보존 수 |
| `--averaged-checkpoint-count` | 3 | Validation 상위 보존 및 평균 checkpoint 수 |

`--device auto`는 가능한 경우 CUDA를 사용하고, `--gpu-id`가 주어지면 해당 GPU를
선택한다.

## 12. 출력 구조

기본 output root는 `DCSPG/checkpoints/fixed_split`이다. `--run-name`을 생략하면 timestamp
기반 폴더가 생성되며, 같은 이름이 이미 있으면 `_001`, `_002` suffix를 붙인다.

```text
run_xxx/
├── run_config.json
├── summary.json
├── last.pt
├── best_checkpoint.pth
├── second_best_checkpoint.pth
├── third_best_checkpoint.pth
├── averaged_checkpoint.pth
├── top_checkpoints.csv
├── train_history.csv
├── validation_support_samples.csv
├── validation_results.csv
├── validation_dataset_summary.csv
├── validation_summary.csv
├── test_support_samples.csv
├── test_results_best_checkpoint.csv
├── test_summary_best_checkpoint.csv
├── test_results_averaged_checkpoint.csv
├── test_summary_averaged_checkpoint.csv
├── checkpoints/
│   └── epoch_XXXX.pth
└── log/
    ├── epoch_metrics.csv
    ├── train_validation_curve.png
    ├── validation_dataset_spearman_curve.png
    ├── validation_weighted_ce.csv
    └── validation_weighted_ce_curve.png
```

`validation_results.csv`는 episode별 생성 수식, support indices, RPN/infix/LaTeX,
Spearman 및 invalid reason을 기록한다. Dataset/epoch별 집계는
`validation_dataset_summary.csv`, 전체 validation criterion 이력은
`validation_summary.csv`에 저장된다.

Test의 `*_results_*.csv`는 repeat별 세부 결과를, `*_summary_*.csv`는 dataset별 및
전체 macro average를 기록한다. `*_results_*.csv`의 `Avg. CE` 열은 각 generated
formula의 10-support-episode 평균 teacher-forcing CE를 기록한다.

## 13. 해석 시 주의점

- Training loss는 teacher formula token imitation 성능이고 validation/test Spearman은
  실제 proxy ranking 품질이다. 두 값은 같은 척도가 아니다.
- Validation과 test는 greedy decoding만 사용한다. `DCSPG/evaluate.py`에 beam search
  기능이 있지만 이 fixed-split training entrypoint에서는 사용하지 않는다.
- Fixed support는 training seed와 무관하지만 모델 학습, train sampling, teacher sampling은
  seed의 영향을 받는다.
- Invalid formula penalty를 바꾸면 checkpoint ranking과 test 평균이 달라질 수 있다.
- `max_formula_len`은 EOS를 포함한다. 너무 작게 설정하면 GroundTruth 로딩 단계에서
  target length 오류가 발생한다.
- Dataset별 teacher formula 수가 다르더라도 16-teacher 모드에서는 formula를 균등
  비복원 sampling하고, 선택된 teacher의 품질 weight는 loss에서 다시 정규화한다.
