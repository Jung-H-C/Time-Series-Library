# `score_candidates.py` 프록시 스코어링 설명서

## 개요

`score_candidates.py`는 `candidates/*.json` 파일을 입력으로 받아, 안에 들어 있는 각 candidate model에 대해 여러 zero-cost proxy를 계산하고 결과를 CSV로 저장하는 스크립트입니다.

엔트리포인트:

```bash
python score_candidates.py --candidates TimesNet_long_term_forecast_ETTh1
```

실제 구현:

- [proxy_scorer.py](/home/gpuadmin/junghc/Time-Series-Library/benchmarking/proxy_scorer.py)


## 기본 철학

이 스크립트는 ESPnet의 [score_proxy.py](/home/gpuadmin/junghc/espnet/egs2/librispeech_100/mamba_practice_asr1/proxy/score_proxy.py)에서 사용한 프록시들을 가능한 한 비슷하게 Time-Series-Library로 옮기는 것을 목표로 합니다.

다만 두 프로젝트는 다음이 다릅니다.

- 모델 구조
- 태스크 종류
- forward signature
- loss 정의
- 공통 encoder API 존재 여부

그래서 **가능한 한 유사하게** 옮기되, Time-Series-Library에서 모든 backbone/task에 공통으로 적용될 수 있도록 일부 로직은 일반화했습니다.


## 전체 동작 흐름

1. `candidates.json` 로드
2. 각 candidate의 `run_args`와 `run.py` 기본값을 합쳐 `args` 생성
3. `task_name`에 맞는 `Exp_*` 클래스를 만들어 모델과 train loader 준비
4. train split에서 minibatch `N`개를 가져옴
5. 각 minibatch에 대해 프록시 점수 계산
6. minibatch 평균을 candidate의 최종 score로 사용
7. 모든 candidate 결과를 CSV로 저장

원하면 `--proxies` 옵션으로 특정 proxy만 골라 계산할 수도 있습니다.

- 예: `--proxies sfrd`
- 예: `--proxies jacob_cov jacob_fro`
- 예: `--proxies sfrd,jacob_cov`

프록시 계산 중에는 stochasticity를 줄이기 위해 다음 정책을 사용합니다.

- BatchNorm 계열은 eval 모드로 전환
- Dropout 계열은 비활성화
- 그 외 모듈은 가능한 한 원래 상태를 유지


## 어떤 batch를 쓰는가

- 기본값은 `5` minibatch입니다
- classification은 `TRAIN`
- 나머지 task는 `train`

즉 실제 학습 루프가 쓰는 train loader에서 앞쪽 minibatch 몇 개를 뽑아 proxy를 계산합니다.


## task별 forward / loss 대응 방식

ESPnet에서는 ASR 모델의 실제 training loss를 써서 grad 기반 proxy를 계산합니다.

Time-Series-Library에서도 같은 원칙을 따르기 위해, 각 task마다 **실제 train loop와 최대한 비슷한 loss**를 만들었습니다.

### 1. `long_term_forecast`, `zero_shot_forecast`

- decoder input 생성
- `model(batch_x, batch_x_mark, dec_inp, batch_y_mark)`
- 마지막 `pred_len` 구간 추출
- `MS`면 마지막 target 차원만 사용
- loss는 `MSE`

### 2. `short_term_forecast`

- 실제 train loop처럼
- `model(batch_x, None, dec_inp, None)`
- `exp._select_criterion(args.loss)` 사용
- `frequency_map`, `batch_y_mark`까지 포함한 원래 loss 형태 유지

### 3. `imputation`

- batch마다 mask를 하나 생성
- masked input을 만들어 model에 입력
- masked position에 대해서만 reconstruction loss 계산

### 4. `anomaly_detection`

- reconstruction output과 원 입력 사이의 `MSE`

### 5. `classification`

- `model(batch_x, padding_mask, None, None)`
- `CrossEntropyLoss`


## 각 프록시 설명

아래는 현재 구현된 프록시들과, ESPnet 원본 대비 어떻게 옮겼는지 설명입니다.

### `params`

- 의미: 총 trainable parameter 수
- 구현: `sum(p.numel() for p in model.parameters())`
- ESPnet과 거의 동일

### `flops`

- 의미: forward 계산량 근사치
- ESPnet 원본: encoder 구조를 분석해서 analytical FLOPs 계산
- Time-Series 버전: 공통 encoder 분석식이 없어서 `torch.profiler(..., with_flops=True)`로 forward FLOPs를 근사 계산

주의:

- profiler가 모든 연산의 FLOPs를 잡아주지는 않습니다
- 따라서 이 값은 ESPnet의 analytical FLOPs와 정확히 같은 성격은 아닙니다
- 그래도 backbone 간 상대 비교용 근사 proxy로는 쓸 수 있습니다

### `grad_norm`

- 의미: 실제 loss backward 후 gradient L2 norm
- 구현: 모든 trainable parameter의 gradient 제곱합의 sqrt
- ESPnet 원본은 encoder parameter 중심이었지만, 여기서는 공통 encoder API가 없어서 **전체 모델 parameter**를 사용합니다

### `snip`

- 의미: `sum(|w * grad|)`
- 구현: backward 후 모든 trainable parameter에 대해 계산
- 원래 SNIP 계열 직관과 동일
- 역시 encoder만이 아니라 전체 모델 기준

### `fisher`

- 의미: activation-level Fisher proxy
- ESPnet 원본: encoder block activation에 hook을 걸고 `(a * grad)^2` 합산
- Time-Series 버전: 공통 encoder block API가 없어서 **파라미터를 직접 갖는 leaf module**에 forward hook을 걸고, 그 activation의 `(a * grad)^2`를 합산

주의:

- 정확히 같은 위치의 activation은 아닙니다
- 대신 모델 내부 활성값을 기반으로 하는 Fisher proxy라는 철학은 유지했습니다

### `grasp`

- 의미: finite-difference Hessian-gradient 기반 GRASP proxy
- 구현:
  - 실제 loss의 gradient 계산
  - 정규화된 방향으로 파라미터를 `+eps`, `-eps` perturb
  - gradient 차이로 Hessian-gradient 근사
  - `-w * Hg` 합산

ESPnet 원본과 가장 유사하게 옮긴 항목 중 하나입니다.

### `jacob_cov`

- 의미: input Jacobian의 sample-wise correlation 구조 기반 proxy
- ESPnet 원본: encoder input 기준 Jacobian
- Time-Series 버전: 공통 encoder input API가 없어서 **모델의 주 입력 `batch_x`**에 대한 Jacobian을 사용
- 구현:
  - `batch_x.requires_grad_(True)`
  - `outputs.sum()`을 pseudo loss로 사용
  - `d outputs / d batch_x` 계산
  - sample별 Jacobian flatten
  - correlation matrix eigenvalue 기반 점수 계산

### `jacob_fro`

- 의미: input Jacobian의 Frobenius norm
- ESPnet 원본과 철학 동일
- Time-Series 버전도 `batch_x`에 대한 gradient를 사용

### `sfrd`

- 의미: 입력 변화량과 representation 변화량의 분리도(discriminability) 계열 proxy
- ESPnet 원본: acoustic feature sequence와 encoder hidden sequence를 비교
- Time-Series 버전:
  - 입력 시계열의 시간 차분 norm
  - 모델 출력 시퀀스의 시간 차분 norm
  - 큰 입력 변화 지점과 작은 입력 변화 지점의 representation change를 비교
- 입력 frame 수와 representation frame 수가 다르면, ESPnet과 마찬가지로 `F.interpolate(..., mode="linear", align_corners=False)`로 길이를 맞춘 뒤 비교

주의:

- 일반적으로는 encoder hidden이 아니라 **모델의 최종 출력 시퀀스**를 representation처럼 사용합니다
- forecast 계열(`long_term_forecast`, `short_term_forecast`, `zero_shot_forecast`)에서는 가능하면 backbone의 raw `forecast()` 출력을 직접 사용합니다
- 즉 train/test loss 계산에 쓰이는 마지막 `pred_len`만의 sliced output보다, `forecast()`가 내놓는 더 긴 전체 sequence representation을 우선 사용합니다
- 다만 classification task에서는 가능한 경우 **classification head 직전의 hidden sequence**를 우선 사용합니다
- 예를 들어 `TimesNet`처럼 마지막에 `[B, seq_len * hidden] -> Linear(num_class)` 형태의 head를 붙이는 모델은, head 입력을 다시 `[B, seq_len, hidden]`으로 reshape해서 SFRD에 사용합니다
- padding mask가 있는 classification batch는 sample별 valid length를 계산해서, 입력 시계열과 hidden representation 모두 유효 길이까지만 잘라서 시간 차분 norm을 계산합니다
- backbone 구조상 raw `forecast()`를 직접 쓰기 어려운 경우에는 기존 forward output으로 fallback 합니다
- 어떤 backbone은 classification head 직전 표현을 sequence 형태로 복원할 수 없어서, 그런 경우에는 기존 fallback 경로를 타거나 `NaN`이 나올 수 있습니다

즉 이 프록시는 ESPnet 원본과 가장 차이가 큰 항목 중 하나입니다.

### `synflow`

- 의미: synflow proxy
- 구현:
  - 모델 파라미터 절댓값화(linearize)
  - 입력을 `ones_like(batch_x)`로 바꿔 forward
  - 출력 합을 pseudo loss로 backward
  - `sum(|w * grad|)`

ESPnet의 synflow 철학을 거의 그대로 일반화한 버전입니다.


## 왜 encoder-only가 아니라 전체 모델을 쓰는가

ESPnet ASR 모델은 `model.encoder`가 비교적 일관된 인터페이스를 가집니다.

반면 Time-Series-Library는 backbone마다 구조가 매우 다릅니다.

- 어떤 모델은 encoder-like 구조가 뚜렷함
- 어떤 모델은 pure MLP/linear
- 어떤 모델은 decomposition block 위주
- 어떤 모델은 foundation model wrapper

그래서 공통으로 비교하려면:

- encoder에만 의존하는 구현은 일부 backbone에서 깨질 수 있고
- 전체 모델 기준으로 프록시를 계산하는 쪽이 더 robust합니다

따라서 현재 구현은 대부분 **전체 모델 기준**으로 일반화했습니다.


## 결과 CSV

기본 출력은:

```text
proxy_scores/<candidates_stem>_proxy_scores.csv
```

특정 proxy subset만 계산하면 기본 출력 파일명도 그에 맞춰 바뀝니다.

```text
proxy_scores/<candidates_stem>_<proxy1>_<proxy2>_proxy_scores.csv
```

컬럼은 기본적으로 다음을 포함합니다.

- `candidate_id`
- `candidate_name`
- `model`
- `task_name`
- `data`
- `num_batches`
- `status`
- `error`
- `params`
- `flops`
- `grad_norm`
- `fisher`
- `grasp`
- `jacob_cov`
- `jacob_fro`
- `sfrd`
- `snip`
- `synflow`


## 해석할 때 주의할 점

이 스크립트의 목적은 **Time-Series-Library 내부에서 서로 다른 candidate를 상대 비교**하는 것입니다.

따라서:

- ESPnet score와 절대값을 직접 비교하면 안 됩니다
- 특히 `flops`, `fisher`, `sfrd`는 구조 차이 때문에 완전히 동일한 정의는 아닙니다
- 대신 “같은 dataset / task / candidate pool 내부의 상대 랭킹” 용도로 쓰는 것이 자연스럽습니다


## 추천 사용법

먼저 1개 candidate만 시험:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --max-candidates 1 \
  --num-batches 5 \
  --gpu-id 0
```

특정 proxy 하나만 계산:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --gpu-id 0 \
  --proxies sfrd
```

여러 proxy만 계산:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --gpu-id 0 \
  --proxies jacob_cov jacob_fro sfrd
```

그 다음 전체 candidate pool 실행:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --gpu-id 0
```

저장 시 candidate row는 항상 `candidate_id`의 마지막 suffix 숫자
예: `_0001`, `_0002`, ..., `_0100`
기준으로 정렬해서 CSV에 기록합니다.


## 추후 네가 검토하면 좋은 부분

특히 아래 항목은 나중에 네가 다시 검토하기 좋습니다.

- `flops`: profiler 기반 근사로 충분한지
- `fisher`: leaf-module activation hook 위치가 적절한지
- `sfrd`: 최종 출력 대신 중간 hidden representation을 써야 더 좋은지
- `jacob_*`: `batch_x` 기준 Jacobian이 각 task에서 가장 적절한지
- foundation model / zero-shot backbone에서도 동일 proxy가 의미 있는지

즉 현재 구현은 **“최대한 유사하게 옮긴 first workable version”**으로 이해하면 가장 정확합니다.
