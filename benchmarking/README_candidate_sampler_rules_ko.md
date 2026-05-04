# `candidate_sampler.py` 규칙 정리

이 문서는 [`benchmarking/candidate_sampler.py`](./candidate_sampler.py) 안에서 `backbone`, `task_name`, `data` 값에 따라 동작이 달라지는 규칙을 한국어로 정리한 문서다.

목적은 다음 두 가지다.

- 나중에 `--sample-search-config`, `--run-candidates`, direct CLI mode를 다시 볼 때 어디서 분기가 일어나는지 빠르게 찾기
- 특히 `classification + UEA`처럼 한 candidate가 실제로는 여러 dataset subset으로 확장 실행되는 예외 동작을 헷갈리지 않기

## 1. 큰 흐름

`candidate_sampler.py`의 동작은 크게 세 단계로 나뉜다.

- 검색 공간을 해석하고 candidate를 샘플링하는 단계
- `backbone/task_name/data` 조합에 맞는 기본 recipe를 찾는 단계
- `--run-candidates` 실행 시 candidate 하나를 실제 `run.py` 호출로 바꾸는 단계

즉, 어떤 값이 샘플링될지는 `search_space`와 backbone 규칙이 결정하고, 어떤 dataset/recipe를 참고할지는 `task_name`과 `data`가 결정하고, 실제 실행 방식이 단일 실행인지 multi-subset sweep인지는 조합별 예외 규칙이 결정한다.

## 2. Backbone별 규칙

현재 `candidate_sampler.py`에서 명시적으로 backbone 전용 규칙을 갖는 것은 `BACKBONE_SPECS`에 등록된 backbone뿐이다.

- 현재 명시적으로 등록된 backbone은 `TimesNet`이다.
- `TimesNet`에 대해서는 아래 파라미터 검증 규칙이 추가로 적용된다.
- `e_layers`: 양의 정수여야 한다.
- `d_model`: 양의 짝수여야 한다.
- `d_ff`: 양의 정수여야 하며, `d_model` 기준 multiplier 형식인 `x2`, `x4` 같은 상대값도 허용된다.
- `top_k`: `task_name`에 따라 상한 계산 방식이 달라진다.
- `num_kernels`: 양의 정수여야 한다.

정리하면, 다른 backbone은 기본적으로 `run.py` 인자 집합을 따르는 일반 규칙만 받고, `TimesNet`만 추가 validator를 가진다.

## 3. Task별 규칙

`task_name`에 따라 달라지는 핵심 규칙은 아래와 같다.

- `TimesNet.top_k` 검증은 `task_name`이 `long_term_forecast`, `short_term_forecast`, `zero_shot_forecast`일 때 `seq_len + pred_len` 기준으로 최대 허용값을 계산한다.
- 그 외 task에서는 `seq_len`만 기준으로 최대 허용값을 계산한다.
- 기본 recipe를 찾을 때는 `examples/<task_name>/...` 아래에 있는 recipe가 더 높은 우선순위를 갖는다.
- direct CLI mode로 spec를 만들 때는 `--fixed task_name=...`를 반드시 줘야 한다.

즉, `task_name`은 단순 메타데이터가 아니라 validation과 recipe 선택 우선순위에 실제로 영향을 준다.

## 4. Data별 규칙

`data` 값에 따라 달라지는 핵심 규칙은 아래와 같다.

- `search_config` 파일명은 기본적으로 `<backbone>_<task_name>_<data>_search_spec.json` 규칙으로 잡힌다.
- direct CLI mode에서는 `--fixed data=...`를 반드시 줘야 한다.
- 기본 recipe를 찾을 때 `run_args.data == data` 조건이 맞는 recipe만 후보가 된다.
- 기본 candidate prefix도 기본적으로 `<backbone>_<task_name>_<data>`를 slugify한 값이 된다.

즉, `data`도 `task_name`과 함께 spec 파일 위치, recipe 선택, candidate naming에 직접 관여한다.

## 5. Backbone + Task + Data 조합 규칙

이 문서에서 가장 중요한 예외 규칙은 아래 한 가지다.

- `task_name == "classification"` 이고 `data == "UEA"`이면 `_candidate_recipe_runs()`가 활성화된다.
- 이 경우 candidate 하나를 단일 `run.py` 호출로 끝내지 않고, 해당 backbone/task/data 조합의 기본 recipe에 들어 있는 모든 run을 순회한다.
- recipe 안의 run 개수가 2개 이상이면 multi-subset sweep로 취급한다.
- 실제 loop는 `_execute_candidate_plan()` 안에서 `recipe_runs` 전체를 순회하며 수행된다.

즉, `classification + UEA`는 `--run-candidates` 시 일반적인 "candidate 1개 = 실행 1번" 규칙을 따르지 않는다.

## 6. `classification + UEA`에서 recipe가 candidate를 어떻게 덮어쓰는가

`classification + UEA` sweep가 시작되면 candidate의 `run_args` 일부는 recipe 쪽 값으로 덮어써진다.

현재 덮어쓰는 키는 아래와 같다.

- `root_path`
- `model_id`
- `batch_size`
- `learning_rate`
- `train_epochs`
- `patience`
- `itr`

의미는 다음과 같다.

- candidate JSON은 backbone과 탐색된 하이퍼파라미터를 담는다.
- 실제 subset별 dataset 경로와 dataset 이름은 recipe가 결정한다.
- 따라서 `timesnet_classification_uea_candidates.json` 안에 `EthanolConcentration`이 들어 있어도, `--run-candidates`에서는 그 값이 subset별 recipe 값으로 바뀌면서 전체 UEA subset sweep가 돌아간다.

## 7. 현재 `TimesNet + classification + UEA`에서 사용되는 기본 recipe

현재 `TimesNet + classification + UEA` 조합의 기본 recipe는 아래 파일로 연결된다.

- [`examples/classification/TimesNet.json`](../examples/classification/TimesNet.json)

이 recipe는 현재 `num_runs = 10`으로 정의되어 있다. 즉, candidate 하나가 아래 10개 subset dataset에 대해 각각 실행된다.

- `EthanolConcentration`
- `FaceDetection`
- `Handwriting`
- `Heartbeat`
- `JapaneseVowels`
- `PEMS-SF`
- `SelfRegulationSCP1`
- `SelfRegulationSCP2`
- `SpokenArabicDigits`
- `UWaveGestureLibrary`

중요한 점은 "10개"라는 숫자가 `candidate_sampler.py` 내부에 직접 하드코딩되어 있는 것이 아니라, 현재 선택된 recipe 파일의 `runs` 개수에서 온다는 것이다.

## 8. 이 recipe는 어디서 만들어지는가

현재 `examples/classification/TimesNet.json`은 아래 스크립트에서 유래한다.

- [`scripts/classification/TimesNet.sh`](../scripts/classification/TimesNet.sh)

즉, 실제 사람이 읽기 쉬운 원본은 shell script이고, `candidate_sampler.py`는 그로부터 생성된 `examples/*.json` recipe를 읽는다.

관련해서 기억하면 좋은 점은 아래와 같다.

- `candidate_sampler.py --refresh-default-recipes` 옵션을 사용하면 `scripts/` 기준으로 `examples/` recipe JSON들을 다시 생성할 수 있다.
- 따라서 나중에 UEA subset 목록을 바꾸고 싶으면 `candidate_sampler.py`보다 먼저 `scripts/classification/TimesNet.sh` 또는 생성된 recipe JSON을 확인하는 편이 맞다.

## 9. Search config 초기화 시의 규칙

spec 파일이 아직 없을 때는 `backbone/task_name/data` 조합을 기준으로 기본 recipe를 찾고, 그 recipe의 첫 run을 참고해서 `fixed_config`를 초기화한다.

여기서 주의할 점은 아래와 같다.

- 초기화에 쓰이는 값은 recipe의 첫 run 기준이다.
- 이때 `model`과 `model_id`는 제외하고 나머지 `run_args`만 `fixed_config`로 들어간다.
- 그래서 `search_config/timesnet_classification_uea_search_spec.json`의 `fixed_config`가 `EthanolConcentration` 기준처럼 보여도, 실제 `--run-candidates` 단계에서는 다시 UEA recipe 전체 run으로 확장될 수 있다.

즉, spec의 `fixed_config`는 "기본 seed run에서 가져온 초기값"이지, `classification + UEA` 실행 시의 최종 dataset 고정값이라고 보면 안 된다.

## 10. 이름 규칙과 연결되는 부분

조합별 규칙은 아니지만, 나중에 결과 폴더명과 연결될 때 자주 헷갈리는 부분이라 같이 적는다.

- 기본 candidate prefix는 `<backbone>_<task_name>_<data>` 형태다.
- candidate 이름은 `candidate_prefix + 탐색 파라미터 토큰 + 4자리 index`로 만들어진다.
- 실행 직전에는 `results_id`가 기본적으로 `candidate_name`으로 채워진다.
- classification 계열에서는 `model_id`를 dataset 식별자로 유지하고, `des`에는 `candidate_name`이 붙는다.

즉, `classification + UEA`에서는 dataset 이름은 subset recipe의 `model_id`가 맡고, candidate 식별은 `candidate_name`과 `results_id`가 맡는 구조다.

## 11. 빠른 판단용 체크리스트

나중에 어떤 candidate가 왜 그런 식으로 실행되는지 헷갈리면 아래 순서로 보면 된다.

- `candidate.run_args.model`, `task_name`, `data`가 무엇인지 본다.
- 이 조합이 `classification + UEA`인지 확인한다.
- 맞다면 `examples/<task>/...json` 중 해당 backbone/data 조합의 기본 recipe가 무엇인지 본다.
- 그 recipe의 `runs` 개수를 본다.
- `runs`가 2개 이상이면 candidate 1개가 recipe의 모든 subset run으로 확장 실행된다.

## 12. 현재 시점 한 줄 요약

현재 저장소 기준으로는 다음처럼 이해하면 된다.

- `TimesNet + classification + UEA`는 `--run-candidates` 시 candidate 하나당 UEA 10개 subset 전체를 실행한다.
- 그 규칙의 분기 시작점은 `benchmarking/candidate_sampler.py`의 `_candidate_recipe_runs()`이다.
- 실제 10개 subset 목록은 `examples/classification/TimesNet.json`과 그 원본인 `scripts/classification/TimesNet.sh`에 있다.
