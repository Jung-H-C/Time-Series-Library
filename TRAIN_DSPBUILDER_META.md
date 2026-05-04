# `train_dspbuilder_meta.py` 사용 설명서

## 개요

`train_dspbuilder_meta.py`는 DSPBuilder용 dataset-conditioned proxy weight generator를 학습하는 스크립트입니다.

현재 학습 방식은 예전의 "dataset 하나를 여러 번 반복한 뒤 다음 dataset으로 넘어가는 메타러닝 스타일"이 아니라, 아래와 같은 supervised mini-batch 방식입니다.

1. `benchmark/*.csv`에서 dataset별 metric / proxy table을 읽습니다.
2. `candidates/*_candidates.json`에서 dataset 로딩 정보를 읽고 실제 train split을 준비합니다.
3. train step마다 train dataset들 중 `train_batch_size`개를 균일분포로 랜덤 샘플링합니다.
4. 샘플링된 각 dataset마다 `support_size`개의 support sample을 뽑아 `dataset_description`을 만듭니다.
5. `dataset_description`으로부터 `weight_vector`를 예측합니다.
6. 각 dataset마다 benchmark candidate를 샘플링해 pair-wise ranking loss를 계산합니다.
7. mini-batch 안의 dataset loss를 평균내서 한 번 weight update를 수행합니다.
8. 이런 update를 `iterations_per_epoch`번 수행하면 1 epoch가 끝납니다.
9. 매 epoch마다 validation을 수행하고, best validation checkpoint를 저장합니다.
10. 학습 종료 후 best checkpoint로 test split을 평가합니다.


## 입력 파일 전제

### 1) Benchmark CSV

`./benchmark/*.csv` 파일을 사용합니다.

- 첫 번째 열: 실제 성능 metric (`mse`, `mase` 등)
- 두 번째 열부터: proxy 값
- 각 row는 candidate model 하나를 의미합니다

proxy 차원 수는 모든 benchmark CSV에서 같아야 합니다.


### 2) Candidate JSON

`./candidates/DSPBuilder_*_candidates.json` 파일을 사용합니다.

이 파일에서 dataset 로딩에 필요한 `fixed_config` 또는 첫 candidate의 `run_args`를 읽어 실제 raw dataset train split을 불러옵니다.


### 3) Raw dataset

각 candidate JSON이 가리키는 `root_path`, `data_path` 위치에 실제 dataset이 준비되어 있어야 합니다.

- 로컬 CSV가 있으면 로컬 파일 사용
- 없으면 repo의 dataset loader fallback 동작 사용


## 실행 위치

프로젝트 루트에서 실행하는 것을 권장합니다.

```bash
cd /data/Time-Series-Library
```


## 기본 사용법

가장 권장되는 방식은 split을 명시적으로 넣는 것입니다.

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Weather,Traffic,ECL,Etth1,Exchange,M4_Hourly,M4_Monthly,M4_Quarterly,M4_Weekly \
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
  --train-datasets Weather,Traffic,ECL,Etth1,Exchange,M4_Hourly,M4_Monthly,M4_Quarterly,M4_Weekly \
  --val-datasets ILI,M4_Daily \
  --test-datasets M4_Yearly \
  --epochs 50 \
  --iterations-per-epoch 200 \
  --train-batch-size 16 \
  --support-size 5 \
  --train-query-size 20 \
  --val-iterations-per-dataset 5 \
  --eval-iterations-per-dataset 5 \
  --device cuda:1
```

### 2) CPU smoke test

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Exchange \
  --val-datasets M4_Daily \
  --test-datasets ILI \
  --epochs 1 \
  --iterations-per-epoch 2 \
  --train-batch-size 2 \
  --val-iterations-per-dataset 1 \
  --eval-iterations-per-dataset 1 \
  --device cpu \
  --output-dir /tmp/dspbuilder_meta_smoke
```

### 3) Proxy signature regression 모드

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Weather,Traffic,ECL,Etth1,Exchange \
  --val-datasets ILI \
  --test-datasets M4_Daily \
  --proxy-signature-regression \
  --cls-loss-weight 0.5
```

### 4) Train / validation만 수행

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py \
  --train-datasets Weather,Traffic,ECL,Etth1,Exchange,M4_Hourly,M4_Monthly,M4_Quarterly,M4_Weekly \
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
- `--val-datasets`
  - validation split dataset 이름 목록
- `--test-datasets`
  - test split dataset 이름 목록
- `--epochs`
  - 최대 epoch 수
  - 기본값: `100`
- `--iterations-per-epoch`
  - train stage에서 epoch당 optimizer update 횟수
  - 기본값: `200`
- `--train-batch-size`
  - train mini-batch에서 한 번에 샘플링할 dataset 개수
  - 기본값: `16`
- `--val-iterations-per-dataset`
  - validation loss 계산 시 dataset당 고정 query batch 수
  - 기본값: `5`
- `--eval-iterations-per-dataset`
  - test stage에서 dataset당 support resampling 반복 횟수
  - 기본값: `5`
- `--support-size`
  - dataset description 생성용 support sample 개수
  - 기본값: `5`
- `--train-query-size`
  - train stage에서 dataset당 샘플링할 candidate 수
  - 기본값: `20`
- `--val-query-size`
  - validation loss 계산 시 고정 query batch 하나의 candidate 수
  - 기본값: `20`
- `--test-query-size`
  - 현재 CLI에는 남아 있지만 `run_test_epoch()`에서는 사용하지 않습니다
  - 현재 test는 매 iteration마다 전체 candidate를 모두 사용합니다
  - 기본값: `10`
- `--encoder-hidden-dim`
  - feature-wise temporal encoder hidden dim
  - 기본값: `16`
- `--hidden-dim`
  - weight head / classifier / signature head 내부 hidden dim
  - 기본값: `32`
- `--weight-head-layers`
  - dataset description에서 weight vector를 만드는 MLP hidden layer 수
  - 기본값: `1`
- `--raw_stat_emb`
  - raw statistic embedding branch 활성화
  - 현재 parser 기본값은 `off`
  - 켜려면 `--raw_stat_emb`, 끄려면 `--no-raw_stat_emb`
- `--dropout`
  - dropout 비율
  - 기본값: `0.1`
- `--learning-rate`
  - optimizer learning rate
  - 기본값: `1e-4`
- `--weight-decay`
  - optimizer weight decay
  - 기본값: `1e-4`
- `--cls-loss-weight`
  - auxiliary loss 가중치
  - dataset classification 또는 proxy signature regression loss에 곱해집니다
  - 기본값: `0.0`
- `--proxy-signature-regression`
  - auxiliary loss를 dataset-id classification 대신 proxy signature regression으로 바꿉니다
  - 기본값: `off`
- `--patience`
  - early stopping patience
  - 기본값: `5`
- `--seed`
  - random seed
  - 기본값: `2026`
- `--device`
  - `auto`, `cpu`, `cuda:0` 등
- `--train-only`
  - train / validation만 수행하고 final test evaluation은 생략합니다
- `--output-dir`
  - run 결과 저장 디렉터리
  - 기본값: `./meta_checkpoints/dspbuilder_meta`


## 학습 로직 요약

### Train stage

train은 supervised mini-batch 방식으로 동작합니다.

- 한 epoch는 `iterations_per_epoch`개의 optimizer step으로 구성됩니다.
- 각 step마다 train dataset 목록에서 `train_batch_size`개를 균일분포로 랜덤 샘플링합니다.
- 이 샘플링은 `rng.choice(tasks)` 기반이라 중복 허용입니다.
- 따라서 train dataset 개수가 `train_batch_size`보다 작아도 정상 동작합니다.
- 샘플링된 각 dataset마다 `support_size`개의 support sample을 `train_dataset`에서 다시 랜덤 샘플링합니다.
- dataset 내 train sample 수가 `support_size`보다 작으면 support sample도 중복 허용 샘플링으로 채웁니다.
- support sample들을 모델에 넣어 `dataset_description`과 `weight_vector`를 만듭니다.
- 해당 dataset benchmark에서 `train_query_size`개 candidate를 랜덤 샘플링하고 pair-wise ranking loss를 계산합니다.
- auxiliary loss는 아래 둘 중 하나입니다.
  - 기본: dataset classification cross-entropy
  - `--proxy-signature-regression` 사용 시: proxy signature regression MSE
- 최종 dataset loss는 `pair_loss_mean + cls_loss_weight * auxiliary_loss` 입니다.
- mini-batch 안의 dataset loss를 평균내고 한 번만 `backward()` / `optimizer.step()`을 수행합니다.
- gradient clipping은 `max_norm=1.0`으로 적용됩니다.
- train terminal 로그는 epoch당 대략 10번 정도 중간 진행상황을 출력합니다.


### Validation stage

validation은 매 epoch 동일한 기준으로 loss를 비교하기 위해 고정 평가 계획을 사용합니다.

validation에는 두 가지 계산이 함께 들어 있습니다.

1. 고정 query batch 기반 validation loss
2. 고정 support set 기반 validation Spearman 분석

#### 1) Validation loss

- 각 valid dataset마다 support index set 하나를 고정합니다.
- `val_query_size * val_iterations_per_dataset`개 candidate를 앞에서부터 가져와 `val_query_size`개씩 고정 분할합니다.
- 예를 들어 기본값이면 `0-19`, `20-39`, `40-59`, `60-79`, `80-99` 식입니다.
- 매 epoch마다 같은 support indices와 같은 query batch들로 pair-wise loss를 계산합니다.
- validation에서는 optimizer가 없으므로 auxiliary loss는 계산되지 않고, loss는 pair-wise ranking loss만 사용합니다.

#### 2) Validation Spearman 분석

- 각 valid dataset마다 서로 다른 support set 여러 개를 미리 고정 샘플링합니다.
- 기본적으로 support set 개수는 `5`개입니다.
- 각 support set으로 weight vector를 만들고, 해당 dataset의 전체 candidate에 대해 Spearman correlation을 계산합니다.
- 이 값은 terminal과 log에 기록되며, `history.json`에는 epoch별 `val_spearman_mean`으로 저장됩니다.

주의:

- validation loss 계산에는 최소 `val_query_size * val_iterations_per_dataset`개의 candidate row가 필요합니다.
- 기본 설정에서는 각 valid benchmark CSV에 최소 `100`개 candidate row가 필요합니다.


### Test stage

- test는 best validation checkpoint를 다시 불러온 뒤 수행합니다.
- 각 test dataset마다 `eval_iterations_per_dataset`번 반복합니다.
- 매 반복마다 해당 dataset의 `train_dataset`에서 `support_size`개 support sample을 랜덤 샘플링합니다.
- support sample로부터 weight vector를 만든 뒤, test에서는 전체 candidate를 모두 score합니다.
- predicted score ranking과 실제 metric ranking 사이의 Spearman correlation을 계산합니다.
- metric이 "낮을수록 좋은 값"이라는 전제 때문에 내부적으로 Spearman 부호를 뒤집어 저장합니다.
- dataset별 `spearman_mean`, `spearman_std`를 기록하고, 전체 dataset / iteration에 대한 overall mean / std도 요약합니다.
- test는 validation처럼 고정 query chunk 계획을 쓰지 않습니다.
- `--train-only`를 켜면 test stage는 수행하지 않습니다.


## 모델 구조

### 1) Sample encoder

`FeatureWiseSharedEncoder`는 sample 하나를 `[time, feature]` 텐서로 입력받습니다.

동작 순서는 다음과 같습니다.

1. feature별 시간축 정규화
2. 각 feature를 공유 Conv1d temporal encoder에 독립적으로 통과
3. time 축 평균 pooling
4. linear projection으로 feature별 32차원 표현 생성
5. feature 축 mean / std pooling으로 64차원 요약 생성
6. `--raw_stat_emb`가 켜져 있으면 raw statistics 8개를 32차원으로 projection
7. 최종 sample embedding 생성

차원은 다음과 같습니다.

- `--no-raw_stat_emb`일 때: `32 + 32 = 64`
- `--raw_stat_emb`일 때: `32 + 32 + 32 = 96`

raw statistics는 아래 8개입니다.

- global mean
- global std
- mean of feature means
- std of feature means
- mean of feature stds
- std of feature stds
- temporal diff mean
- temporal diff std


### 2) Set encoder와 dataset description

support sample마다 sample embedding을 만든 뒤, 이를 `SetEncoder`로 집계해 하나의 `dataset_description`을 만듭니다.

- 입력 shape: `[num_support_samples, sample_embedding_dim]`
- 출력 shape: `[dataset_description_dim]`
- 현재 `dataset_description_dim` 기본값: `128`

즉 예전처럼 support embedding들의 mean / std를 바로 이어붙이는 구조가 아니라, MLP 기반 set encoder가 support set 전체를 요약합니다.


### 3) Heads

`dataset_description`에서 아래 head들이 갈라집니다.

- `weight_head`
  - 출력: `weight_vector`
  - shape: `[proxy_dim]`
  - 마지막에 `tanh`가 적용되어 각 원소는 `(-1, 1)` 범위입니다
- `dataset_classifier`
  - 출력: `dataset_logits`
  - shape: `[num_dataset_classes]`
- `signature_head`
  - 출력: `predicted_signature`
  - shape: `[proxy_dim]`

실제 ranking에는 `weight_vector`가 직접 사용됩니다.


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
  - epoch별 train / validation 평균 통계
- `summary.json`
  - best epoch, best checkpoint, final test 결과 요약
- `best_checkpoint.pth`
  - validation loss 기준 best model
- `checkpoint.pth`
  - `EarlyStopping` 내부 저장 checkpoint
- `train_logs/<dataset>.txt`
  - train 단계에서 샘플링된 dataset별 iteration 로그
- `valid_logs/<dataset>.txt`
  - validation loss / Spearman 로그
- `test_logs/<dataset>.txt`
  - test iteration 로그

`--train-only`인 경우에는 `test_logs/`가 생성되지 않고, `summary.json`의 test 관련 값은 `null`로 저장됩니다.


## 로그 형식

### Train / validation loss 로그

dataset별 `.txt` 파일에는 아래 항목들이 기록됩니다.

- `epoch`
- `dataset`
- `iteration`
- `loss`
- `pair_acc`
- `pair_loss_mean`
- `cls_loss`
- `dataset_acc`
- `num_pairs`
- `weight_norm`
- `weight_vector`

auxiliary loss가 proxy signature regression일 때는 로그에 `signature_cosine`도 추가로 기록됩니다.

주의:

- train 로그의 `iteration`은 "dataset 순번"이 아니라 "epoch 내 batch step 번호"입니다.
- 같은 train iteration 안에 여러 dataset 로그가 생길 수 있습니다.


### Validation 추가 로그

validation 로그 파일 상단에는 아래 정보가 남습니다.

- `fixed_loss_support_indices`
- `fixed_spearman_support_indices`
- `fixed_query_ranges`

그리고 validation Spearman 분석 결과가 같은 파일 뒤쪽에 함께 이어집니다.


### Test 로그

test iteration 로그에는 아래 항목들이 기록됩니다.

- `epoch`
- `dataset`
- `metric`
- `iteration`
- `spearman_corr`
- `num_candidates`
- `support_indices`
- `weight_norm`
- `weight_vector`

validation / test의 Spearman summary 로그와 terminal 출력에는 `benchmark/lookup/spearman_baseline.csv`에 baseline이 있으면 아래 비교 정보도 함께 출력됩니다.

- `baseline_best_proxy`
- `baseline_coefficient`


## Early Stopping / Best Checkpoint

- validation loss가 감소하면 `best_checkpoint.pth`를 저장합니다
- `patience` epoch 동안 개선이 없으면 early stopping 합니다
- 학습 종료 후 best checkpoint를 다시 불러와 test split을 평가합니다

중요:

- best model 선택 기준은 validation Spearman이 아니라 validation loss입니다
- validation Spearman은 모니터링 및 분석용으로 별도 기록됩니다


## 자주 확인할 점

### 1) dataset split은 겹치면 안 됩니다

같은 dataset을 train / val / test에 동시에 넣으면 에러가 발생합니다.

### 2) 모든 benchmark CSV는 같은 proxy 차원을 가져야 합니다

현재 스크립트는 모든 dataset이 동일한 proxy column 구성을 가진다고 가정합니다.

### 3) valid dataset은 candidate 수가 충분해야 합니다

validation loss 계산에는 최소 `val_query_size * val_iterations_per_dataset`개 candidate가 필요합니다.

### 4) test는 현재 전체 candidate ranking 평가입니다

`--test-query-size` 인자가 있더라도 현재 구현에서는 test에서 subset query sampling을 하지 않습니다.

### 5) PyTorch 환경에서 실행해야 합니다

권장 예시:

```bash
conda run -n tslib_nightly python train_dspbuilder_meta.py ...
```


## 추천 시작점

처음에는 작은 split과 짧은 epoch로 smoke test를 먼저 돌린 뒤, checkpoint / history / log 파일이 정상적으로 쌓이는지 확인한 다음 본 실험으로 넘어가는 것을 권장합니다.
