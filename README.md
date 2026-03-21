<중요>

search_config/ : 이 경로안에 실험에 사용할 backbone + task + data 정의 파일(.json)이 저장됨 (.json파일은  examples/ 경로에 있는 레시피 코드 참조해서 작성하면 됨 )

candidates/ : 이 경로안에 각 search space별 candidate models정의 파일(.json)이 정의됨

result_<task>.txt : 이 경로안에 model test summary가 append형식으로 저장됨

proxy_scores/ : 이 경로안에 proxy score가 저장됨

<명령어>

sample_candidates.py 

1. 저장된 search config를 바탕으로 후보를 N개 샘플링함

```bash
python sample_candidates.py \
  --sample-search-config TimesNet_long_term_forecast_ETTh1 \
  --num-samples N
```

2. 후보 JSON을 실제 실행(train+test)합니다.

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --gpu-id 0
```

여러 GPU에 병렬 실행하려면:

```bash
python sample_candidates.py \
  --run-candidates TimesNet_long_term_forecast_ETTh1 \
  --gpu-id 0 1 2 3
```

`--run-candidates`는 candidate별로 고유한 실행 namespace가 되도록 내부적으로 `des`를 분리해서 `checkpoints/`, `results/`, `test_results/` 충돌을 막습니다. `model_id`는 그대로 유지합니다.

`classification + UEA` candidate를 실행하면, 기본 recipe(`examples/classification/<Backbone>.json`)에 들어 있는 여러 subset을 모두 순회합니다. 예를 들어 `TimesNet`은 10개 subset accuracy를 모두 모아서 candidate별 평균 accuracy를 `results/<candidates_stem>_uea_average_accuracy.csv`에 저장합니다.

score_candidiates.py

PROXY_SCORER.md 파일 참고

<별로 안 중요함>
results/ : 이 경로안에 model test 결과가 저장됨

test_results/ : 이 경로안에 model test 결과 중 몇개를 sampling해서 visualize한 figure가 저장됨
