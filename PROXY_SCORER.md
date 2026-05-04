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

기본적으로 BatchNorm 계열은 `eval` 모드, Dropout 계열은 `off` (`eval`) 로 세팅되어 있습니다.
재현가능한 실험 결과를 얻기 위해 `--deterministic` 옵션 사용을 권장합니다.

`--gpu-id`는 다음 형식을 모두 지원합니다.

- `--gpu-id 0`
- `--gpu-id 0 1 2`
- `--gpu-id 0,1,2`

GPU를 여러 개 주면 GPU 개수만큼 worker를 띄워 candidate들을 병렬로 score합니다.
즉 `--gpu-id 0 1 2`를 주면 `cuda:0`, `cuda:1`, `cuda:2`를 각각 담당하는 3개의 worker가 동시에 실행됩니다.


## 전체 동작 흐름

1. `candidates.json` 로드
2. 각 candidate의 `run_args`와 `run.py` 기본값을 합쳐 `args` 생성
3. `task_name`에 맞는 `Exp_*` 클래스를 만들어 모델과 train loader 준비
4. train split에서 minibatch `N`개를 가져옴
5. 각 minibatch에 대해 프록시 점수 계산
6. 기본 동작에서는 minibatch 평균을 candidate의 최종 score로 사용
7. 모든 candidate 결과를 CSV로 저장

원하면 `--proxies` 옵션으로 특정 proxy만 골라 계산할 수도 있습니다.

- 예: `--proxies sfrd`
- 예: `--proxies jacob_cov jacob_fro`
- 예: `--proxies sfrd,jacob_cov`
`sfrd`는 (`long_term_forecast`에서) 입력/예측 출력 변화 벡터의 frame-wise cosine similarity 평균으로 계산됩니다.

`--deterministic` 옵션을 주면 seed 고정뿐 아니라 PyTorch deterministic mode까지 함께 켭니다.
즉 cuDNN autotuning을 끄고, 가능한 경우 결정론적인 알고리즘만 사용하도록 강제합니다.

프록시 계산 중에는 stochasticity를 줄이기 위해 다음 정책을 사용합니다.

- BatchNorm 계열은 eval 모드로 전환
- Dropout 계열은 비활성화
- 그 외 모듈은 가능한 한 원래 상태를 유지
- `--deterministic` 사용 시 PyTorch deterministic algorithms 활성화

원하면 `--proxy-bn-mode train` 옵션으로 BatchNorm 계열만 train 모드로 유지한 채
proxy를 계산할 수 있습니다. Dropout 계열은 여전히 eval 모드로 비활성화됩니다.


## 어떤 batch를 쓰는가

- 기본값은 `5` minibatch입니다
- classification은 `TRAIN`
- 나머지 task는 `train`
- proxy scoring용 train loader를 별도로 만들고, `shuffle=True` 상태에서 seed로 순서를 고정합니다
- 즉 candidate마다 같은 **랜덤 셔플 순서**를 공유하고, 그 순서에서 앞 `N`개 minibatch를 사용합니다
- seed도 candidate마다 다시 고정해서, model init과 batch 선택이 candidate 순서에 덜 의존하게 했습니다
- 실행 간 수치 재현성이 중요하면 `--deterministic`을 함께 사용하는 것이 좋습니다

즉 실제 학습에 사용하는 train split에서, **seed로 고정된 shuffled order**의 앞쪽 minibatch 몇 개를 뽑아 proxy를 계산합니다.

`--separate` 옵션을 주면 여기서 평균을 내지 않고, 샘플링한 각 minibatch의 proxy 결과를 **배치별 row**로 그대로 CSV에 저장합니다.
즉 `--num-batches 5 --separate`이면 candidate당 5개의 row가 생깁니다.
candidate가 여러 `run_args`로 확장되는 경우(예: UEA subset별 run)는 `run_index`, `run_name`, `batch_index` 컬럼으로 각 row를 구분합니다.


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

- 의미: 입력 변화량과 예측 출력 변화량의 방향 정렬도를 보는 proxy
- Time-Series 버전:
  - `long_term_forecast`에서만 계산
  - 입력 시계열과 모델 예측 출력에서 frame-wise 변화 벡터(`x[t]-x[t-1]`, `y_hat[t]-y_hat[t-1]`)를 만듦
  - 각 frame에서 두 변화 벡터의 cosine similarity를 계산하고 평균
  - 최종 score는 이 평균값으로, 이론적으로 `[-1, 1]` 범위

주의:

- 입력 변화 벡터 길이와 예측 출력 변화 벡터 길이가 다르면 해당 배치의 `sfrd`는 `NaN`으로 처리합니다
- 입력/예측 feature 차원이 다르면 해당 배치의 `sfrd`는 `NaN`으로 처리합니다

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

`--separate`를 주면 기본 출력 파일명은 아래처럼 바뀝니다.

```text
proxy_scores/<candidates_stem>_separate_proxy_scores.csv
```

특정 proxy subset만 계산하면 기본 출력 파일명도 그에 맞춰 바뀝니다.

```text
proxy_scores/<candidates_stem>_<proxy1>_<proxy2>_proxy_scores.csv
```

`--separate`와 같이 쓰면:

```text
proxy_scores/<candidates_stem>_<proxy1>_<proxy2>_separate_proxy_scores.csv
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

`--separate`를 켜면 메타 컬럼이 아래처럼 바뀝니다.

- `candidate_id`
- `candidate_name`
- `model`
- `task_name`
- `data`
- `run_index`
- `run_name`
- `batch_index`
- `num_batches`
- `status`
- `error`

여기서:

- `run_index`는 candidate 내부에서 몇 번째 run인지 나타내는 1-based index입니다
- `run_name`은 보통 해당 run의 `model_id`입니다
- `batch_index`는 샘플링된 minibatch의 1-based index입니다
- `num_batches`는 그 run에서 샘플링한 전체 batch 수입니다


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

배치별 개별 결과를 저장하고 싶다면:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --separate \
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

`sfrd`만 계산:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --gpu-id 0 \
  --proxies sfrd
```

재현성을 더 강하게 맞추고 싶다면:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --gpu-id 0 \
  --proxies sfrd \
  --deterministic
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

여러 GPU로 병렬 실행하고 싶다면:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --gpu-id 0 1 2
```

또는 쉼표 형식도 가능합니다:

```bash
python score_candidates.py \
  --candidates-file candidates/timesnet_long_term_forecast_etth1_candidates.json \
  --num-batches 5 \
  --gpu-id 0,1,2
```

저장 시 row는 우선 `candidate_id`의 마지막 suffix 숫자
예: `_0001`, `_0002`, ..., `_0100`
기준으로 정렬해서 CSV에 기록합니다.

`--separate`를 켠 경우에는 같은 candidate 안에서 `run_index`, `batch_index` 순서로 추가 정렬됩니다.


## 추후 네가 검토하면 좋은 부분

특히 아래 항목은 나중에 네가 다시 검토하기 좋습니다.

- `flops`: profiler 기반 근사로 충분한지
- `fisher`: leaf-module activation hook 위치가 적절한지
- `sfrd`: 현재 representation 선택 heuristic이 backbone 전반에 충분히 잘 맞는지
- `jacob_*`: `batch_x` 기준 Jacobian이 각 task에서 가장 적절한지
- foundation model / zero-shot backbone에서도 동일 proxy가 의미 있는지

즉 현재 구현은 **“최대한 유사하게 옮긴 first workable version”**으로 이해하면 가장 정확합니다.
