# `run_candidate_benchmark_pipeline.py` 사용 설명서

## 개요

`run_candidate_benchmark_pipeline.py`는 아래 3단계를 한 번에 수행하는 파이프라인 스크립트입니다.

1. `candidates.json`의 모든 candidate를 실제로 train/test 실행
2. 모든 candidate에 대해 10개 zero-cost proxy score 계산
3. train/test metric(`mse`)과 proxy score를 합쳐
   - raw benchmark CSV
   - rank-normalized benchmark CSV
   를 생성


## 실행 위치

프로젝트 루트에서 실행하는 것을 권장합니다.

```bash
cd /data/Time-Series-Library
```


## 기본 사용법

```bash
python run_candidate_benchmark_pipeline.py \
  --candidates-file candidates/your_candidates.json \
  --gpu-id 0
```

멀티 GPU 예시:

```bash
python run_candidate_benchmark_pipeline.py \
  --candidates-file candidates/your_candidates.json \
  --gpu-id 0 1 2 3 \
  --run-workers-per-gpu 1
```

proxy 계산 재현성을 높이고 싶을 때:

```bash
python run_candidate_benchmark_pipeline.py \
  --candidates-file candidates/your_candidates.json \
  --gpu-id 0 \
  --num-batches 5 \
  --seed 2026 \
  --proxy-deterministic
```


## 인자 설명

- `--candidates-file` (필수)
  - 실행할 candidate JSON 경로
- `--gpu-id`
  - 사용할 물리 GPU id 목록
  - 예: `--gpu-id 0`, `--gpu-id 0 1`, `--gpu-id 0,1,2`
  - `run-candidates`와 `proxy scoring` 양쪽에 동일하게 사용
- `--run-workers-per-gpu` (기본값: `1`)
  - `sample_candidates.py --run-candidates-file` 단계에서 GPU당 worker 수
- `--num-batches` (기본값: `5`)
  - proxy 계산 시 랜덤 샘플링할 mini-batch 개수
- `--seed` (기본값: `2026`)
  - proxy batch 샘플링 시드
- `--output-dir` (기본값: `benchmark_results`)
  - 출력 CSV 저장 디렉터리
- `--proxy-deterministic`
  - proxy 계산을 deterministic 모드로 실행
- `--continue-on-error`
  - run-candidates 단계에서 일부 candidate 실패 시에도 다음 candidate 계속 진행


## 내부 동작 순서

### 1) Candidate train/test 실행

내부적으로 아래 명령을 호출합니다.

```bash
python sample_candidates.py --run-candidates-file <candidates_file> ...
```

이 단계 결과로 각 run의 `results/.../metrics.npy`가 생성되어야 다음 단계에서 metric을 수집할 수 있습니다.


### 2) 10개 proxy 계산

내부적으로 아래 명령을 호출합니다.

```bash
python score_candidates.py --candidates-file <candidates_file> ...
```

사용되는 proxy 컬럼은 고정 10개입니다.

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


### 3) metric + proxy 머지 및 정규화

- `metrics.npy`에서 candidate별 metric 평균을 만듭니다.
  - 최종 머지/정규화 CSV에는 `mse`만 사용
- Step2 proxy CSV와 `candidate_id` 기준으로 머지합니다.
- 각 proxy 컬럼에 대해 rank-based normalization을 수행합니다.

정규화 규칙:

- 값이 클수록 높은 순위(내림차순)
- tie는 평균 rank 사용
- 정규화 식: `(N - rank) / (N - 1)`
- 따라서 최고 rank는 `1`, 최저 rank는 `0`
- 값이 비어있거나 `NaN`이면 정규화 결과도 `NaN`


## 출력 파일

기본적으로 `--output-dir` 아래에 timestamp가 붙은 CSV들을 생성합니다.

1. `<candidates_stem>_training_metrics_raw_<timestamp>.csv`
   - run 단위 raw metric 수집 결과
2. `<candidates_stem>_proxy_scores_raw_<timestamp>.csv`
   - proxy 점수 raw CSV 복사본
3. `<candidates_stem>_benchmark_raw_with_mse_<timestamp>.csv`
   - `mse` + raw proxy를 candidate 단위로 합친 파일
4. `<candidates_stem>_benchmark_ranknorm_with_mse_<timestamp>.csv`
   - `mse` + rank-normalized proxy를 candidate 단위로 합친 파일


## 주요 상태 컬럼 해석

- `training_status`
  - `success`: 모든 run에서 metric 수집 성공
  - `partial`: 일부 run만 성공
  - `failed`: 모든 run 실패
- `proxy_status`, `proxy_error`
  - `score_candidates.py` 결과 상태/오류 전달


## 참고

- UEA처럼 subset recipe가 있는 candidate JSON은 내부적으로 subset run들을 전개해서 metric을 집계합니다.
- `metrics.npy`가 없거나 깨져 있으면 해당 run은 실패로 기록되고, candidate 집계는 가능한 값만 평균냅니다.
- proxy CSV에 필수 proxy 컬럼이 하나라도 없으면 파이프라인은 오류로 중단됩니다.
