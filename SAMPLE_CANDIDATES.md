# `sample_candidates.py` 사용 설명서

## 개요

`sample_candidates.py`는 `benchmarking/candidate_sampler.py`를 감싸는 CLI 엔트리포인트입니다.

이 도구는 크게 다섯 가지 작업을 지원합니다.

1. 사용 가능한 백본과 하이퍼파라미터를 확인
2. `search_config/` 아래에 search space 설정 파일 생성 또는 갱신
3. 저장된 search config를 기반으로 후보 모델 `N`개를 랜덤 샘플링
4. `candidates/` 아래 후보 JSON을 불러와 실제 `run.py` 학습/테스트를 순차 실행
5. `scripts/` 아래의 기본 shell recipe를 JSON으로 변환하여 `examples/`를 재생성


## 디렉터리 역할

- `search_config/`
  - 랜덤 샘플링에 사용할 search space 설정 파일을 저장합니다.
  - 파일 이름 규칙: `<backbone>_<task>_<dataset>_search_spec.json`

- `candidates/`
  - 샘플링된 후보 모델 JSON 파일을 저장합니다.
  - 파일 이름 규칙: 보통 `<search_config_stem>_candidates.json`

- `examples/`
  - `scripts/` 아래 기본 학습 shell recipe를 JSON으로 변환해 저장합니다.
  - 디렉터리 구조는 `scripts/`를 그대로 따라갑니다.
  - 이 파일들은 참고용 기본 recipe이며, search space 샘플링용 설정 파일은 아닙니다.


## 기본 실행 위치

모든 명령은 프로젝트 루트에서 실행하는 것을 권장합니다.

```bash
cd /home/gpuadmin/junghc/Time-Series-Library
```


## 1. 사용 가능한 백본 목록 확인

```bash
python sample_candidates.py --list-backbones
```


## 2. 백본별 하이퍼파라미터 확인

특정 모델 파일이 실제로 사용하는 `configs.*` 인자들을 출력합니다.

```bash
python sample_candidates.py --describe-backbone TimesNet
python sample_candidates.py --describe-backbone TimeMixer
python sample_candidates.py --describe-backbone PatchTST
```

모든 백본에 대해 한 번에 확인하려면:

```bash
python sample_candidates.py --describe-all-backbones
```


## 3. Search Config 생성 또는 갱신

direct CLI 모드에서는 `search_config/` 아래의 파일을 생성하거나 업데이트합니다.

중요:

- direct CLI 모드에서 `backbone`, `task_name`, `data`는 필수입니다.
- 즉 `--backbone ...`, `--fixed task_name=...`, `--fixed data=...`를 반드시 함께 넣어야 합니다.
- 이 셋 중 하나라도 없으면 search config 생성은 실패합니다.
- 새 search config를 처음 만들 때는 `run.py` 기본값으로 시작하지 않고, `examples/` 아래에서 같은 `backbone/task_name/data` 조합의 기본 recipe를 찾아 그 `run_args`를 초기 fixed setting으로 사용합니다.
- `examples/`에서 일치하는 기본 recipe를 찾지 못하면 search config는 생성되지 않습니다.

예시:

```bash
python sample_candidates.py \
  --backbone TimesNet \
  --fixed task_name=long_term_forecast \
  --fixed data=ETTh1 \
  --fixed root_path=./dataset/ETT-small/ \
  --fixed data_path=ETTh1.csv \
  --fixed features=M \
  --fixed freq=h \
  --fixed seq_len=96 \
  --fixed label_len=48 \
  --fixed pred_len=96 \
  --fixed enc_in=7 \
  --fixed dec_in=7 \
  --fixed c_out=7 \
  --search e_layers=1,2,3,4 \
  --search d_model=64,128,256,512 \
  --search d_ff=x2,x4,x6 \
  --search top_k=2,3,4,5,6,7,8 \
  --search num_kernels=3,4,5,6,7,8
```

이 명령은 아래 파일을 생성 또는 갱신합니다.

```text
search_config/timesnet_long_term_forecast_etth1_search_spec.json
```

설명:

- `--fixed key=value`
  - `run.py` 인자 중 고정할 값을 지정합니다.
  - 여러 번 반복해서 사용할 수 있습니다.

- `--search key=v1,v2,v3`
  - 특정 하이퍼파라미터의 후보 리스트를 지정합니다.
  - 같은 key가 기존 `fixed_config`에 있으면, spec 저장 시 그 key는 `fixed_config`에서 제거되고 `search_space`만 유지됩니다.
  - 여러 번 반복해서 사용할 수 있습니다.

- 같은 key를 한 명령에서 `--fixed`와 `--search`에 동시에 넣을 수는 없습니다.

- direct CLI 모드에서 `--search-spec-name`을 주면:
  - 자동 생성된 `<backbone>_<task>_<dataset>_search_spec.json` 대신, 지정한 파일명으로 `search_config/` 아래에 저장합니다.
  - `.json` 확장자를 생략하면 자동으로 붙습니다.

- direct CLI 모드에서 `--candidates-name`을 주면:
  - `--output`을 직접 쓰지 않아도, 샘플링 결과를 지정한 파일명으로 `candidates/` 아래에 저장합니다.
  - `.json` 확장자를 생략하면 자동으로 붙습니다.

- direct CLI 모드에서 `--output`을 생략하면:
  - search config만 갱신하고 종료합니다.
  - 후보 샘플링은 수행하지 않습니다.

- 새 search config가 처음 생성될 때:
  - `examples/`의 matching recipe에서 `model`, `model_id`를 제외한 `run_args`를 `fixed_config`의 초기값으로 채웁니다.
  - 선택된 기본 recipe 정보는 search config의 `default_recipe` 필드에 같이 저장됩니다.


## 4. 저장된 Search Config로 후보 모델 샘플링

저장된 config를 `backbone_task_dataset` 키로 불러와 샘플링할 수 있습니다.

```bash
python sample_candidates.py \
  --sample-search-config TimesNet_long_term_forecast_ETTh1 \
  --num-samples 20
```

이 명령은 기본적으로 아래 파일을 생성합니다.

```text
candidates/timesnet_long_term_forecast_etth1_candidates.json
```

특정 search config 파일을 직접 지정할 수도 있습니다.

```bash
python sample_candidates.py \
  --sample-search-config-file search_config/timesnet_long_term_forecast_etth1_search_spec.json \
  --num-samples 20
```


원하면 출력 경로를 직접 지정할 수도 있습니다.

```bash
python sample_candidates.py \
  --sample-search-config TimesNet_long_term_forecast_ETTh1 \
  --num-samples 20 \
  --output candidates/my_timesnet_candidates.json
```


## 5. Search Config 갱신과 샘플링을 한 번에 수행

direct CLI 모드에서 `--output`까지 함께 주면, 아래 순서로 동작합니다.

1. `search_config/` 아래의 해당 search config 파일 갱신
2. 후보 모델 샘플링 수행
3. `candidates/` 또는 지정한 출력 경로에 결과 저장

예시:

```bash
python sample_candidates.py \
  --backbone TimeMixer \
  --fixed task_name=long_term_forecast \
  --fixed data=ETTh1 \
  --search e_layers=1,2,3 \
  --search d_model=64,128 \
  --num-samples 6 \
  --output candidates/timemixer_long_term_forecast_etth1_candidates.json
```

파일명만 간단히 지정하고 싶다면:

```bash
python sample_candidates.py \
  --backbone TimesNet \
  --fixed task_name=classification \
  --fixed data=UEA \
  --search e_layers=1,2,3 \
  --search d_model=32,64 \
  --num-samples 8 \
  --search-spec-name timesnet_classification_uea_custom \
  --candidates-name timesnet_classification_uea_candidates_custom
```

위 명령은 아래 두 파일을 사용합니다.

```text
search_config/timesnet_classification_uea_custom.json
candidates/timesnet_classification_uea_candidates_custom.json
```


## 6. Candidate JSON을 실제로 순차 실행

샘플링된 후보 JSON을 불러와, 안에 들어 있는 각 `run_args`를 사용해 `run.py`를 순차 실행할 수 있습니다.

`candidates/` 아래 파일 이름을 기준으로 실행하려면:

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --gpu-id 0
```

여러 GPU에 병렬 분산하려면:

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --gpu-id 0 1 2 3
```

또는 쉼표 형태도 가능합니다.

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --gpu-id 0,1,2,3
```

이 명령은 아래 파일을 읽습니다.

```text
candidates/timesnet_long_term_forecast_etth1_candidates.json
```

특정 candidate JSON 파일을 직접 지정할 수도 있습니다.

```bash
python sample_candidates.py \
  --run-candidates-file candidates/timemixer_long_term_forecast_etth1_candidates.json \
  --gpu-id 1
```

또는 `candidates/` 아래 파일명만 직접 지정할 수도 있습니다.

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --candidates-name my_timesnet_candidates.json \
  --gpu-id 0
```

이 경우 `--run-candidates`의 key 대신 실제로는 아래 파일을 읽습니다.

```text
candidates/my_timesnet_candidates.json
```

설명:

- 실행 namespace 처리
  - `--run-candidates` 실행 시에는 각 candidate가 고유한 실행 namespace를 쓰도록 내부적으로 `des`를 재작성합니다.
  - 기본적으로 `<원래 des>__<candidate_name>` 형태를 사용합니다.
  - `model_id`는 그대로 유지하므로 classification처럼 `model_id`를 dataset 이름으로 쓰는 경우도 안전합니다.
  - 그래서 멀티 GPU 병렬 실행 시에도 `checkpoints/`, `results/`, `test_results/` 경로가 서로 충돌하지 않습니다.
  - 추가로 `classification + UEA` candidate JSON은 기본 recipe인 `examples/classification/<Backbone>.json`의 모든 subset run을 따라갑니다.
  - 예를 들어 `TimesNet` UEA classification candidate는 10개 subset을 모두 학습/테스트하고, candidate별 평균 accuracy를 `results/<candidates_stem>_uea_average_accuracy.csv`에 저장합니다.

- `--gpu-id`
  - 물리 GPU 번호를 하나 이상 지정합니다.
  - 예: `--gpu-id 3`, `--gpu-id 0 1 2 3`, `--gpu-id 0,1,2,3`
  - GPU를 하나만 주면 기존처럼 순차 실행합니다.
  - GPU를 여러 개 주면 GPU마다 worker 1개씩 띄우고, 비는 GPU에 다음 candidate를 붙여 병렬 실행합니다.
  - 각 worker는 내부적으로 `CUDA_VISIBLE_DEVICES=<gpu-id>`를 설정하고 `run.py`에는 `--gpu 0`을 넘깁니다.
  - 즉 물리 GPU 3번을 쓰고 싶으면 `--gpu-id 3`으로 주면 되고, `--gpu-id 0 1 2 3`이면 총 4개 GPU에 분산됩니다.

- `--dry-run`
  - 실제 학습은 시작하지 않고, 어떤 `run.py` 명령이 순차 실행될지만 출력합니다.

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --gpu-id 0 \
  --dry-run
```

- `--continue-on-error`
  - 특정 후보가 실패해도 뒤 후보들을 계속 실행합니다.
  - 이 옵션이 없으면 첫 실패 지점에서 중단합니다.
  - 단, 멀티 GPU 병렬 실행 중이면 이미 다른 GPU에서 돌고 있던 in-flight job은 끝까지 실행될 수 있습니다.


## 7. JSON Spec 파일을 직접 사용하는 모드

직접 준비한 JSON 파일을 `--spec`으로 넘겨 사용할 수도 있습니다.

```bash
python sample_candidates.py \
  --spec search_config/timesnet_long_term_forecast_etth1_search_spec.json \
  --output candidates/timesnet_long_term_forecast_etth1_candidates.json
```

`--spec` 모드에서는 `--output`이 반드시 필요합니다.


## 8. `examples/` 아래 기본 Recipe 재생성

`examples/`는 `scripts/` 아래 shell recipe를 JSON으로 변환한 결과물입니다.

전체를 다시 생성하려면:

```bash
python sample_candidates.py --refresh-default-recipes
```

이 명령은 예를 들어 아래와 같은 파일들을 생성합니다.

```text
examples/long_term_forecast/ETT_script/TimesNet_ETTh1.json
examples/classification/MambaSL.json
examples/anomaly_detection/MSL/Autoformer.json
```


## Search Config JSON 형식

대표적인 형식은 아래와 같습니다.

```json
{
  "backbone": "TimesNet",
  "num_samples": 10,
  "seed": 2026,
  "candidate_prefix": "timesnet_long_term_forecast_etth1",
  "fixed_config": {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "data": "ETTh1",
    "root_path": "./dataset/ETT-small/",
    "data_path": "ETTh1.csv",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7
  },
  "search_space": {
    "e_layers": [1, 2, 3, 4],
    "d_model": [64, 128, 256, 512],
    "d_ff": ["x2", "x4", "x6"],
    "top_k": [2, 3, 4, 5, 6, 7, 8],
    "num_kernels": [3, 4, 5, 6, 7, 8]
  }
}
```


## Candidate Output JSON 형식

샘플링 결과 JSON은 크게 두 부분으로 구성됩니다.

- `metadata`
  - backbone
  - seed
  - 요청한 샘플 개수
  - 전체 유효 조합 수
  - fixed config
  - search space

- `candidates`
  - `candidate_id`
  - `candidate_name`
  - `model`
  - `hyperparameters`
  - `run_args`

여기서 `run_args`는 이후 `run.py` 실행 파이프라인에 연결하기 쉽게 저장된 인자 dict입니다.


## 지원하는 값 입력 방식

### 스칼라 값

```bash
--search d_model=64,128,256
--search dropout=0.1,0.2,0.3
--search activation=gelu,relu
```

### 상대값

예를 들어 TimesNet처럼 `d_ff`를 `d_model`에 상대적으로 지정하고 싶을 때:

```bash
--search d_ff=x2,x4,x6
```

이 경우 `d_ff`는 기준 파라미터 값을 기준으로 계산됩니다.

### JSON 리스트 값

리스트 형태의 fixed config를 넣고 싶을 때:

```bash
--fixed p_hidden_dims='[128,128]'
```


## 알아두면 좋은 점

- 기본적으로 중복 없이(unique) 샘플링합니다.
- `num_samples`가 가능한 조합 수보다 크면:
  - `num_samples`를 줄이거나
  - `--allow-replacement`를 사용해야 합니다.
- 일부 백본은 추가적인 validation 규칙을 가집니다.
  - 현재는 TimesNet 쪽 validation이 가장 구체적으로 들어가 있습니다.
- 백본 목록은 `models/` 디렉터리에서 자동 탐색됩니다.


## 권장 사용 흐름

1. 먼저 백본의 사용 가능 인자를 확인합니다.

```bash
python sample_candidates.py --describe-backbone TimesNet
```

2. search config를 생성하거나 갱신합니다.

```bash
python sample_candidates.py \
  --backbone TimesNet \
  --fixed task_name=long_term_forecast \
  --fixed data=ETTh1 \
  --search e_layers=1,2,3,4 \
  --search d_model=64,128,256,512
```

3. 저장된 search config를 바탕으로 후보를 샘플링합니다.

```bash
python sample_candidates.py \
  --sample-search-config TimesNet_long_term_forecast_ETTh1 \
  --num-samples 20
```

4. 후보 JSON을 실제 실행합니다.

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --gpu-id 0
```

5. 필요하면 기본 학습 recipe를 `examples/`에서 참고합니다.

```text
examples/long_term_forecast/ETT_script/TimesNet_ETTh1.json
```
