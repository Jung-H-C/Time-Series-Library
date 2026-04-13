# `train_dspbuilder_meta.py` 사용 설명서

## 개요

`train_dspbuilder_meta.py`는 DSPBuilder 메타러닝 학습 스크립트입니다.

이 스크립트는 아래 흐름으로 동작합니다.

1. `benchmark/*.csv`를 lookup dictionary처럼 읽어 dataset별 metric/proxy table 구성
2. `candidates/*_candidates.json`의 `fixed_config`를 읽어 각 dataset의 실제 train split 로드
3. 각 dataset에서 mini sample 5개를 뽑아 dataset embedding 생성
4. embedding으로부터 10차원 proxy weight vector 생성
5. benchmark CSV에서 candidate를 샘플링해 pair-wise ranking loss 계산
6. train set 전체를 한 바퀴 돌면 validation 수행
7. validation 평균 loss가 3 epoch 이상 개선되지 않으면 early stopping
8. best validation checkpoint 저장 후 test set 평가


## 입력 파일 전제

### 1) Benchmark CSV

`./benchmark/*.csv` 파일을 사용합니다.

- 파일명 예시: `DSPBuilder_Weather_Benchmark.csv`
- 첫 번째 열: 실제 성능 metric (`mse`, `mase` 등)
- 두 번째 열부터: 10개 centered rank-normalized proxy 값

즉 각 row는 candidate model 하나를 의미합니다.


### 2) Candidate JSON

`./candidates/DSPBuilder_*_candidates.json` 파일을 사용합니다.

이 파일에서 dataset 로딩에 필요한 `fixed_config` 또는 첫 candidate의 `run_args`를 읽어,
실제 raw dataset train split에서 mini sample을 샘플링합니다.


### 3) Raw dataset

각 candidate JSON이 가리키는 `root_path`, `data_path` 위치에 실제 dataset이 준비되어 있어야 합니다.

- 로컬 CSV가 있으면 로컬 파일 사용
- 없으면 repo의 dataset loader fallback 동작을 따름


## 실행 위치

프로젝트 루트에서 실행하는 것을 권장합니다.

```bash
cd /data/Time-Series-Library
```


## 기본 사용법

가장 권장되는 방식은 split을 명시적으로 넣는 것입니다.

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Weather,Traffic,ECL,Etth1,Exchange \
  --val-datasets ILI,M4_Daily \
  --test-datasets M4_Yearly
```

인자를 생략하면 실행 중 prompt로 train/val/test dataset 이름을 입력받습니다.

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py
```


## 예시 명령어

### 1) 기본 학습

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Weather,Traffic,ECL,Etth1,Exchange,M4_Hourly,M4_Monthly,M4_Quarterly \
  --val-datasets ILI,M4_Daily \
  --test-datasets M4_Yearly \
  --epochs 50 \
  --iterations-per-dataset 5 \
  --val-iterations-per-dataset 10 \
  --eval-iterations-per-dataset 5 \
  --support-size 5 \
  --train-query-size 10 \
  --val-query-size 10 \
  --test-query-size 10 \
  --device cuda:1
```

### 2) CPU smoke test

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Exchange \
  --val-datasets M4_Daily \
  --test-datasets ILI \
  --epochs 1 \
  --iterations-per-dataset 1 \
  --val-iterations-per-dataset 10 \
  --eval-iterations-per-dataset 1 \
  --device cpu \
  --output-dir /tmp/dspbuilder_meta_smoke
```

### 3) Train/validation만 수행

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Weather,Traffic,ECL,Etth1,Exchange \
  --val-datasets ILI,M4_Daily \
  --train-only \
  --device cuda:0
```

이 경우 test dataset 입력은 필요하지 않으며, early stopping 또는 max epoch 종료 후 test 평가 없이 종료합니다.


## 주요 인자 설명

- `--benchmark-dir`
  - benchmark CSV 디렉터리
  - 기본값: `./benchmark`
- `--candidate-dir`
  - candidate JSON 디렉터리
  - 기본값: `./candidates`
- `--train-datasets`
  - train split dataset 이름 목록
  - 쉼표 또는 공백으로 구분
- `--val-datasets`
  - validation split dataset 이름 목록
- `--test-datasets`
  - test split dataset 이름 목록
- `--epochs`
  - 최대 epoch 수
  - 기본값: `50`
- `--iterations-per-dataset`
  - train stage에서 dataset당 update 반복 횟수
  - 기본값: `5`
- `--val-iterations-per-dataset`
  - validation stage에서 dataset당 평가 반복 횟수
  - 기본값: `10`
- `--eval-iterations-per-dataset`
  - test stage에서 dataset당 평가 반복 횟수
  - 기본값: `5`
- `--support-size`
  - dataset embedding용 mini sample 개수
  - 기본값: `5`
- `--train-query-size`
  - train stage 각 iteration에서 샘플링할 candidate 수
  - 기본값: `10`
- `--val-query-size`
  - validation stage 고정 chunk 하나당 candidate 수
  - 기본값: `10`
- `--test-query-size`
  - test stage 각 iteration에서 샘플링할 candidate 수
  - 기본값: `10`
- `--encoder-hidden-dim`
  - feature-wise shared encoder 내부 hidden dim
  - 기본값: `64`
- `--raw_stat_emb`
  - encoder의 raw statistic embedding branch on/off 옵션
  - 기본값: `on`
  - 끄려면 `--no-raw_stat_emb`
  - `on`이면 encoder output이 `96`, task embedding이 `192`
  - `off`이면 encoder output이 `64`, task embedding이 `128`
- `--hidden-dim`
  - final MLP hidden dim
  - 기본값: `128`
- `--dropout`
  - MLP dropout
  - 기본값: `0.1`
- `--learning-rate`
  - optimizer learning rate
  - 기본값: `1e-3`
- `--weight-decay`
  - optimizer weight decay
  - 기본값: `1e-4`
- `--patience`
  - early stopping patience
  - 기본값: `3`
- `--seed`
  - random seed
  - 기본값: `2026`
- `--device`
  - `auto`, `cpu`, `cuda:0` 등
- `--train-only`
  - train/validation만 수행하고 final test evaluation은 생략
  - 이 옵션을 쓰면 `--test-datasets`는 필요하지 않음
- `--output-dir`
  - run 결과 저장 디렉터리
  - 기본값: `./meta_checkpoints/dspbuilder_meta`


## 학습 로직 요약

### Train stage

- 매 epoch 시작 전에 `train_tasks` 순서를 shuffle합니다.
- 각 dataset마다 `support_size=5` mini sample을 한 번 랜덤 샘플링합니다.
- 이렇게 뽑은 support sample 5개는 그 dataset의 `iterations-per-dataset` 동안 동일하게 유지됩니다.
- 이 고정된 5개 sample로 10차원 proxy weight vector를 반복 계산합니다.
- benchmark CSV에서 `train_query_size`개 candidate를 랜덤 샘플링합니다.
- 가능한 모든 pair `train_query_size * (train_query_size - 1) / 2`개에 대해 pair-wise loss를 계산합니다.
- dataset당 `iterations-per-dataset`번 update 후 다음 dataset으로 넘어갑니다.
- terminal에는 train stage에 대해서만 dataset별 평균 loss가 출력됩니다.


### Validation stage

validation은 매 epoch 동일한 기준으로 loss를 비교하기 위해 고정 평가 규칙을 사용합니다.

- 각 valid dataset마다 support sample 5개를 한 번만 고정
- candidate `val_query_size * val_iterations_per_dataset`개를 `val_query_size`개씩 고정 분할
- 예를 들어 기본값이면 `0-9`, `10-19`, ..., `90-99`
- 따라서 매 epoch마다 같은 support sample과 같은 candidate chunk 순서로 validation loss를 계산

주의:

- 기본 설정에서는 valid dataset benchmark CSV에 최소 `100`개 candidate row가 있어야 합니다.
- 일반적으로는 `val-iterations-per-dataset * val-query-size` 이상 candidate row가 필요합니다.


### Test stage

- test는 `--eval-iterations-per-dataset`만큼 평가합니다.
- 현재 test는 validation처럼 고정 chunk 계획을 쓰지 않습니다.
- `--train-only`를 켜면 test stage는 아예 수행하지 않습니다.


## Encoder / Weight Head 구조

`FeatureWiseSharedEncoder`는 sample 하나를 `[time, feature]` 형태로 입력받습니다.

1. feature-wise normalization
2. feature별 temporal encoder 적용
3. time 축 global average pooling
4. linear projection으로 feature별 32차원 표현 생성
5. feature 축 mean pooling 32차원 + std pooling 32차원 계산
6. `--raw_stat_emb`가 켜져 있으면 sample 원본에서 raw statistics 8개를 추출
7. `--raw_stat_emb`가 켜져 있으면 raw statistics를 32차원 raw stat embedding으로 projection
8. `--raw_stat_emb`가 켜져 있으면 `32 + 32 + 32`를 concat하여 최종 sample embedding 96차원 생성
9. `--no-raw_stat_emb`이면 raw stat branch 없이 `32 + 32`만 사용하여 sample embedding 64차원 생성

현재 raw statistics는 아래 값을 사용합니다.

- global mean
- global std
- mean of feature means
- std of feature means
- mean of feature stds
- std of feature stds
- temporal diff mean
- temporal diff std

그 다음:

- raw stat branch가 켜져 있으면 sample 5개의 96차원 embedding에 대해 sample 축 mean/std를 계산하고 dataset embedding 192차원 생성
- raw stat branch가 꺼져 있으면 sample 5개의 64차원 embedding에 대해 sample 축 mean/std를 계산하고 dataset embedding 128차원 생성
- MLP + `tanh`를 통과시켜 최종 10차원 proxy weight vector 생성


## 출력 파일

각 run은 `--output-dir` 아래에 timestamp 디렉터리를 생성합니다.

예시:

```text
meta_checkpoints/dspbuilder_meta/dspbuilder_meta_YYYYMMDD_HHMMSS/
```

생성 파일:

- `config.json`
  - 실행 인자와 split 정보
- `history.json`
  - epoch별 train/validation 평균 통계
- `summary.json`
  - best epoch, best checkpoint, final test 결과 요약
- `best_checkpoint.pth`
  - validation 평균 loss 기준 best model
- `checkpoint.pth`
  - `EarlyStopping` 내부 저장 checkpoint
- `train_logs/<dataset>.txt`
  - train iteration 로그
- `valid_logs/<dataset>.txt`
  - validation iteration 로그
- `test_logs/<dataset>.txt`
  - test iteration 로그

`--train-only`인 경우에는 `test_logs/`가 생성되지 않고, `summary.json`의 test 관련 값은 `null`로 저장됩니다.


## 로그 형식

각 iteration마다 dataset별 `.txt` 파일에 아래 정보가 기록됩니다.

- `epoch`
- `dataset`
- `iteration`
- `loss`
- `pair_acc`
- `pair_loss_mean`
- `num_pairs`
- `weight_norm`
- `weight_vector`

terminal에는 train stage에 대해서만 아래 요약 로그가 출력됩니다.

- `epoch`
- `dataset`
- `avg_loss_over_<iterations>_iters`

validation 로그 파일 상단에는 추가로 아래 정보가 남습니다.

- `fixed_loss_support_indices`
- `fixed_spearman_support_indices`
- `fixed_query_ranges`

validation/test의 Spearman summary 로그와 terminal 출력에는, `benchmark/lookup/spearman_baseline.csv`에 baseline이 있으면 아래 비교 정보도 함께 출력됩니다.

- `baseline_best_proxy`
- `baseline_coefficient`


## Early Stopping / Best Checkpoint

- validation 평균 loss가 감소하면 `best_checkpoint.pth` 저장
- `patience=3` 동안 개선이 없으면 early stopping
- 학습 종료 후 best checkpoint를 다시 불러와 test set 평가


## 자주 확인할 점

### 1) dataset split은 겹치면 안 됩니다

같은 dataset을 train/val/test에 동시에 넣으면 에러가 발생합니다.

### 2) 모든 benchmark CSV는 같은 proxy 차원을 가져야 합니다

현재 스크립트는 모든 dataset이 동일한 10개 proxy column을 가진다고 가정합니다.

### 3) valid dataset은 candidate 수가 충분해야 합니다

기본 validation 규칙을 그대로 쓰면 각 valid benchmark CSV에 최소 100개 candidate가 필요합니다.

### 4) PyTorch 환경에서 실행해야 합니다

권장 예시:

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py ...
```


## 추천 시작점

처음에는 작은 split과 짧은 epoch로 smoke test를 먼저 돌린 뒤,
정상적으로 checkpoint/log가 쌓이는 것을 확인하고 본 실험으로 넘어가는 것을 권장합니다.
