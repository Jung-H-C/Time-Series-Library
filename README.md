<중요>

search_config/ : 이 경로안에 실험에 사용할 backbone + task + data 정의 파일(.json)이 저장됨 (.json파일은  examples/ 경로에 있는 레시피 코드 참조해서 작성하면 됨 )

candidates/ : 이 경로안에 각 search space별 candidate models정의 파일(.json)이 정의됨

result_<task>.txt : 이 경로안에 model test summary가 append형식으로 저장됨

proxy_scores/ : 이 경로안에 proxy score가 저장됨

<hr>

## Dataset Integration Checklist

기준:
- `[x]` : 현재 코드베이스에서 **전용 loader / registry 경로가 명시적으로 연결된 경우**
- `[ ]` : 아직 **전용 integration이 없는 경우**
- `custom` CSV로 우회해서 읽을 수 있는 경우가 있더라도, 여기서는 **전용 dataset 지원 여부만** 표시함

| Status | Dataset | 현재 상태 / 메모 |
| --- | --- | --- |
| `[x]` | M1 | 구현됨. `Dataset_MonashTSFGeneric`, `m1_yearly` / `m1_quarterly` / `m1_monthly` short-term 경로 연결 |
| `[x]` | M3 | 구현됨. `Dataset_MonashTSFGeneric`, `m3_yearly` / `m3_quarterly` / `m3_monthly` / `m3_other` short-term 경로 연결 |
| `[x]` | M4 | 구현됨. `Dataset_M4`, `short_term_forecast` 경로 연결 |
| `[x]` | Tourism | 구현됨. `Dataset_TourismMonthlyTSF`, `tourism_monthly` monthly short-term 경로 연결 |
| `[x]` | CIF 2016 | 구현됨. `Dataset_MonashTSFGeneric`, `cif_2016` short-term 경로 연결 |
| `[x]` | London Smart Meters | 구현됨. `Dataset_MonashTSFGeneric`, `london_smart_meters` short-term 경로 연결 |
| `[x]` | Aus. Electricity Demand | 구현됨. `Dataset_MonashTSFGeneric`, `aus_electricity_demand` short-term 경로 연결 |
| `[x]` | Wind Farms | 구현됨. `Dataset_MonashTSFGeneric`, `wind_farms` short-term 경로 연결 |
| `[x]` | Dominick | 구현됨. `Dataset_DominickPanel` + `Dataset_DominickTSF` 지원 |
| `[x]` | Bitcoin | 구현됨. `Dataset_MonashTSFGeneric`, `bitcoin` short-term 경로 연결 |
| `[x]` | Pedestrian Counts | 구현됨. `Dataset_MonashTSFGeneric`, `pedestrian_counts` short-term 경로 연결 |
| `[x]` | Vehicle Trips | 구현됨. `Dataset_MonashTSFGeneric`, `vehicle_trips` short-term 경로 연결 |
| `[x]` | KDD Cup 2018 | 구현됨. `Dataset_MonashTSFGeneric`, `kdd_cup_2018` short-term 경로 연결 |
| `[x]` | Weather | 구현됨. `Dataset_MonashTSFGeneric`, `weather_tsf` short-term 경로 연결 |
| `[x]` | NN5 | 구현됨. `Dataset_NN5DailyTSF`, `nn5_daily` daily short-term 경로 연결 |
| `[x]` | Web Traffic | 구현됨. `Dataset_WebTrafficTSF`, `short_term_forecast` + calendar mark 지원 |
| `[x]` | Solar | 구현됨. `Dataset_MonashTSFGeneric`, `solar_10min` / `solar_weekly` short-term 경로 연결 |
| `[x]` | Electricity | 구현됨. `Dataset_MonashTSFGeneric`, `electricity_hourly` / `electricity_weekly` short-term 경로 연결 |
| `[x]` | Car Parts | 구현됨. `Dataset_CarPartsTSF`, `car_parts` monthly multivariate short-term 경로 연결 |
| `[x]` | FRED-MD | 구현됨. `Dataset_MonashTSFGeneric`, `fred_md` short-term 경로 연결 |
| `[x]` | San Francisco Traffic | 구현됨. `Dataset_MonashTSFGeneric`, `san_francisco_traffic_hourly` / `san_francisco_traffic_weekly` short-term 경로 연결 |
| `[x]` | Rideshare | 구현됨. `Dataset_MonashTSFGeneric`, `rideshare` short-term 경로 연결 |
| `[x]` | Hospital | 구현됨. `Dataset_MonashTSFGeneric`, `hospital` short-term 경로 연결 |
| `[x]` | COVID Deaths | 구현됨. `Dataset_MonashTSFGeneric`, `covid_deaths` short-term 경로 연결 |
| `[x]` | Temperature Rain | 구현됨. `Dataset_MonashTSFGeneric`, `temperature_rain` short-term 경로 연결 |
| `[x]` | Sunspot | 구현됨. `Dataset_MonashTSFGeneric`, `sunspot` short-term 경로 연결 |
| `[x]` | Saugeen River Flow | 구현됨. `Dataset_MonashTSFGeneric`, `saugeen_river_flow` short-term 경로 연결 |
| `[x]` | US Births | 구현됨. `Dataset_MonashTSFGeneric`, `us_births` short-term 경로 연결 |
| `[x]` | Solar Power | 구현됨. `Dataset_MonashTSFGeneric`, `solar_power` short-term 경로 연결 |
| `[x]` | Wind Power | 구현됨. `Dataset_MonashTSFGeneric`, `wind_power` short-term 경로 연결 |

<hr>

### 현재 구현된 항목 요약

- `[x]` `M4`
  - `short_term_forecast` 전용 loader 연결
- `[x]` `Dominick`
  - `dominick`, `dominik` : raw retail panel용 short-term dataset
  - `dominick_tsf` : 체크인된 TSF inspection 경로
- `[x]` `Tourism`
  - `tourism_monthly`, `tourism` : monthly competition TSF용 short-term dataset
  - fixed window + month-of-year calendar `x_mark / y_mark`
- `[x]` `NN5`
  - `nn5_daily`, `nn5` : daily ATM cash-withdrawal TSF용 short-term dataset
  - fixed window + day-of-week/month calendar `x_mark / y_mark`
- `[x]` `Car Parts`
  - `car_parts`, `carparts` : monthly intermittent-demand TSF용 short-term dataset
  - 2674개 car-part series를 하나의 aligned multivariate matrix로 쌓아서 `D=2674` 전체 채널 예측
  - fixed window + month-of-year calendar `x_mark / y_mark`
- `[x]` `Web Traffic`
  - `web_traffic`, `webtraffic`, `kaggle_web_traffic`
  - daily page-level fixed window + real calendar-based `x_mark / y_mark`
- `[x]` `Generic Monash TSF`
  - 이번에 남은 forecastingdata / Monash 계열은 `Dataset_MonashTSFGeneric`으로 1차 통합함
  - 기본 task는 전부 `short_term_forecast`
  - 첫 구현은 각 TSF row를 독립 univariate series로 보고 `past L -> future P` fixed window를 만듦
  - 로컬 파일이 없으면 설정된 Zenodo record에서 artifact를 받아와 `dataset/<alias>/` 아래에 내려받도록 구현함
  - `x_mark / y_mark`는 generic compatibility zero mark를 사용하고, absolute calendar mark가 꼭 필요한 데이터셋은 나중에 bespoke loader로 분리 가능

<hr>

### 이번에 추가한 generic Monash 데이터셋

| Dataset | Alias | 권장 task | 기본 해석 |
| --- | --- | --- | --- |
| `M1` | `m1_yearly`, `m1_quarterly`, `m1_monthly` | `short_term_forecast` | competition series별 univariate forecast |
| `M3` | `m3_yearly`, `m3_quarterly`, `m3_monthly`, `m3_other` | `short_term_forecast` | competition series별 univariate forecast |
| `CIF 2016` | `cif_2016` | `short_term_forecast` | monthly banking series forecast |
| `London Smart Meters` | `london_smart_meters` | `short_term_forecast` | meter별 energy series forecast |
| `Aus. Electricity Demand` | `aus_electricity_demand` | `short_term_forecast` | half-hourly demand series forecast |
| `Wind Farms` | `wind_farms` | `short_term_forecast` | wind-farm series forecast |
| `Bitcoin` | `bitcoin` | `short_term_forecast` | crypto series forecast |
| `Pedestrian Counts` | `pedestrian_counts` | `short_term_forecast` | counter별 hourly series forecast |
| `Vehicle Trips` | `vehicle_trips` | `short_term_forecast` | trip-related weekly series forecast |
| `KDD Cup 2018` | `kdd_cup_2018` | `short_term_forecast` | air-quality / env series forecast |
| `Weather` | `weather_tsf` | `short_term_forecast` | weather station series forecast |
| `Solar` | `solar_10min`, `solar_weekly` | `short_term_forecast` | solar generation series forecast |
| `Electricity` | `electricity_hourly`, `electricity_weekly` | `short_term_forecast` | customer load series forecast |
| `FRED-MD` | `fred_md` | `short_term_forecast` | macroeconomic monthly series forecast |
| `San Francisco Traffic` | `san_francisco_traffic_hourly`, `san_francisco_traffic_weekly` | `short_term_forecast` | sensor/location traffic series forecast |
| `Rideshare` | `rideshare` | `short_term_forecast` | rideshare-related series forecast |
| `Hospital` | `hospital` | `short_term_forecast` | hospital monthly series forecast |
| `COVID Deaths` | `covid_deaths` | `short_term_forecast` | regional daily death series forecast |
| `Temperature Rain` | `temperature_rain` | `short_term_forecast` | climate series forecast |
| `Sunspot` | `sunspot` | `short_term_forecast` | long univariate sunspot forecast |
| `Saugeen River Flow` | `saugeen_river_flow` | `short_term_forecast` | daily river-flow forecast |
| `US Births` | `us_births` | `short_term_forecast` | daily births forecast |
| `Solar Power` | `solar_power` | `short_term_forecast` | ultra-high-frequency solar power forecast |
| `Wind Power` | `wind_power` | `short_term_forecast` | ultra-high-frequency wind power forecast |

<hr>

### 구현된 데이터셋 task 기준

현재 추가한 데이터셋들은 모두 `short_term_forecast` 기준으로 맞춰져 있음.

`short_term_forecast`로 둔 이유:
- 여러 개의 독립 시계열을 같은 방식으로 평가하는 competition / benchmark 성격이 강함
- 목표가 긴 역사 전체를 모델링하는 것보다, 정해진 과거 window로 가까운 미래 horizon을 맞히는 구조에 가까움
- fixed window 또는 competition horizon 형태로 `past L -> future P`를 만들기 쉬움

| Dataset | 권장 task | 한 시계열 단위 | 입력 의미 | 예측 target | mark 의미 |
| --- | --- | --- | --- | --- | --- |
| `m4` | `short_term_forecast` | M4 개별 series | 과거 univariate 값 | 미래 competition horizon | 실제 time mark가 아니라 insample/outsample mask |
| `dominick`, `dominik` | `short_term_forecast` | `(store_id, sku_id)` | 과거 주별 `[sales, price, margin, promo]` | 미래 `sales` | zero mark. interface 호환용 |
| `dominick_tsf` | `short_term_forecast` inspection용 | TSF `series_name` | 과거 주별 단변량 값 | 미래 단변량 값 | placeholder mark |
| `tourism_monthly`, `tourism` | `short_term_forecast` | Tourism 개별 monthly series | 과거 월별 tourism 값 | 미래 월별 tourism 값 | `month_sin`, `month_cos` |
| `nn5_daily`, `nn5` | `short_term_forecast` | ATM 개별 series | 과거 일별 cash withdrawal 값 | 미래 일별 cash withdrawal 값 | `dow_sin`, `dow_cos`, `month_sin`, `month_cos` |
| `car_parts`, `carparts` | `short_term_forecast` | 전체 car-parts monthly matrix | 과거 월별 2674개 part sales vector | 미래 월별 2674개 part sales vector | `month_sin`, `month_cos` |
| `web_traffic`, `webtraffic`, `kaggle_web_traffic` | `short_term_forecast` | Wikipedia page | 과거 일별 traffic 값 | 미래 일별 traffic 값 | `dow_sin`, `dow_cos`, `month_sin`, `month_cos` |
| `Dataset_MonashTSFGeneric` 계열 | `short_term_forecast` | TSF row 하나 | 과거 단변량 value | 미래 단변량 value | zero compatibility mark |

<hr>

### Input feature dimension 기준

주의할 점은 benchmark 표의 `Multivariate = Yes`와 우리 loader의 실제 `D > 1`이 항상 같은 뜻은 아니라는 것임. 원본 표에서 multivariate는 “같은 calendar에 여러 관련 series가 있음”을 뜻하는 경우가 많고, 우리는 구현할 때 두 방식 중 하나를 선택할 수 있음.

- row-wise univariate 방식: 각 series를 독립 sample로 보고 `D=1`로 학습함. NN5, Web Traffic처럼 series 수가 많고 각 entity가 독립적인 경우 첫 구현이 단순하고 안정적임.
- true multivariate 방식: 같은 calendar를 공유하는 여러 series를 channel로 쌓아 `D=series_count`로 학습함. Car Parts처럼 길이가 짧고 모든 series가 같은 monthly grid일 때 더 자연스럽게 멀티 입력을 만들 수 있음.

현재 구현 기준:

| Dataset | 현재 loader의 실제 D | D 해석 | 비고 |
| --- | ---: | --- | --- |
| `m4` | 1 | 단변량 value | 기존 M4 competition loader 유지 |
| `dominick`, `dominik` | 4 | `sales`, `price`, `margin`, `promo` | 진짜 feature multivariate. target은 미래 `sales` 1채널 |
| `dominick_tsf` | 1 | 단변량 value | TSF inspection용 |
| `tourism_monthly`, `tourism` | 1 | 단변량 tourism value | calendar mark는 feature D가 아니라 별도 `M=2` |
| `nn5_daily`, `nn5` | 1 | ATM별 단변량 cash value | 원본은 관련 series 묶음이지만 현재 loader는 row-wise univariate |
| `car_parts`, `carparts` | 2674 | 각 car part series가 하나의 channel | true multivariate, 전체 channel을 동시에 예측 |
| `web_traffic`, `webtraffic`, `kaggle_web_traffic` | 1 | page별 단변량 traffic value | 원본은 관련 page series 묶음이지만 현재 loader는 row-wise univariate |
| `Dataset_MonashTSFGeneric`로 붙인 나머지 Monash 계열 | 1 | 기본적으로 TSF row별 단변량 value | 첫 구현은 generic 안정성을 우선해서 row-wise univariate |

README에 listup된 데이터셋 중 multivariate로 확장 여지가 큰 항목:

| Dataset | 원본 표 기준 multivariate 가능성 | 우리 구현 상태 |
| --- | --- | --- |
| `NN5` | Yes | 구현됨. 현재는 `D=1` row-wise univariate |
| `Web Traffic` | Yes | 구현됨. 현재는 `D=1` page-wise univariate |
| `Car Parts` | Yes | 구현됨. 현재는 `D=2674` true multivariate |
| `Solar` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 site/panel별 true multivariate로 확장 가능 |
| `Electricity` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 고객/계량기별 true multivariate로 확장 가능 |
| `FRED-MD` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 경제 변수들을 함께 쓰는 true multivariate로 확장 가능 |
| `San Francisco Traffic` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 sensor/location별 true multivariate로 확장 가능 |
| `Rideshare` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 지역/서비스별 true multivariate로 확장 가능 |
| `Hospital` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 category/region별 true multivariate로 확장 가능 |
| `COVID Deaths` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 지역별 true multivariate로 확장 가능 |
| `Temperature Rain` | Yes | 구현됨. 현재는 generic `D=1` row-wise univariate, 추후 weather variable/location별 true multivariate로 확장 가능 |

<hr>

### 구현된 데이터셋 예시 sequence

#### M4

M4는 fixed sliding window보다는 기존 repo의 M4 competition loader 방식을 유지함. `seasonal_patterns`별 horizon이 자동으로 정해지고, `seq_len=2*pred_len`, `label_len=pred_len`로 세팅됨.

Monthly 예시:

```text
P = 18
L = 36
label_len = 18

batch_x      = (B, 36, 1)
batch_y      = (B, 36, 1)   # label_len + P
batch_x_mark = (B, 36, 1)   # insample mask
batch_y_mark = (B, 36, 1)   # outsample mask
output       = (B, 18, 1)
```

의미:
- `batch_x`: 모델이 보는 과거 단변량 값
- `batch_y`: 앞 `label_len`은 decoder warm-up, 뒤 `P`는 평가할 미래 값
- `batch_x_mark`, `batch_y_mark`: calendar feature가 아니라 padding/missing 여부를 나타내는 mask

#### Dominick raw panel

Dominick raw panel은 `(store_id, sku_id)` 하나를 하나의 주별 sales series로 봄. 입력은 sales만이 아니라 가격/마진/프로모션을 같이 넣는 multivariate forecast 구조임.

기본 예시:

```text
L = 16
P = 8
label_len = 8
D = 4  # sales, price, margin, promo
M = 1  # zero compatibility mark

seq_x        = (16, 4)
seq_y        = (8, 1)
seq_x_mark   = (16, 1)
seq_y_mark   = (8, 1)

batch_x      = (B, 16, 4)
batch_y      = (B, 8, 1)
batch_x_mark = (B, 16, 1)
batch_y_mark = (B, 8, 1)
output       = (B, 8, 1)
```

시계열 한 구간은 이렇게 해석함:

```text
X_i = [x_i, ..., x_{i+15}]
x_t = [sales_t, price_t, margin_t, promo_t]

Y_i = [sales_{i+16}, ..., sales_{i+23}]
```

의미:
- `sales`: 예측 대상이면서 과거 input feature로도 사용
- `price`: 판매량에 영향을 줄 수 있는 가격 정보
- `margin`: 판매/가격 구조를 보조하는 수익성 정보
- `promo`: 행사 여부를 나타내는 binary feature
- `x_mark`, `y_mark`: 첫 버전에서는 실제 calendar signal이 아니라 모델 interface를 맞추기 위한 zero tensor

#### Dominick TSF

현재 체크인된 `dominick_dataset.tsf`는 raw panel의 `price`, `margin`, `promo`를 포함하지 않는 단변량 TSF inspection 경로임. 그래서 feature-rich Dominick 실험은 `dominick`, `dominik` raw panel loader가 기준이고, `dominick_tsf`는 데이터 규모/shape 확인용에 가까움.

README의 TSF 요약 커맨드 기준 예시:

```text
L = 16
P = 8
label_len = 8
D = 1

seq_x      = (16, 1)
seq_y      = (16, 1)  # label_len + P
seq_x_mark = (16, M)
seq_y_mark = (16, M)
```

#### Tourism Monthly

Tourism Monthly는 개별 tourism monthly series 하나를 한 시계열로 봄. 월별 관광 수요는 연간 계절성이 강하므로 month-of-year mark를 실제 signal로 넣음.

기본 예시:

```text
L = 48
P = 24
label_len = 24
D = 1
M = 2  # month_sin, month_cos

seq_x        = (48, 1)
seq_y        = (24, 1)
seq_x_mark   = (48, 2)
seq_y_mark   = (24, 2)

batch_x      = (B, 48, 1)
batch_y      = (B, 24, 1)
batch_x_mark = (B, 48, 2)
batch_y_mark = (B, 24, 2)
output       = (B, 24, 1)
```

시계열 한 구간은 이렇게 해석함:

```text
X_i = [v_i, ..., v_{i+47}]
Y_i = [v_{i+48}, ..., v_{i+71}]
```

의미:
- `v_t`: 해당 월의 tourism value
- `month_sin`, `month_cos`: 1월과 12월이 숫자상 멀어 보이지 않게 월 정보를 원형 좌표로 표현한 feature
- 값은 train split 통계로 scaling되므로 예시 window에서는 음수/소수가 나올 수 있음

#### NN5 Daily

NN5 Daily는 영국 ATM별 일별 cash withdrawal series 하나를 한 시계열로 봄. ATM 출금량은 요일 효과가 강하고 월별 패턴도 있을 수 있으므로 day-of-week/month calendar mark를 실제 signal로 넣음.

기본 예시:

```text
L = 112
P = 56
label_len = 56
D = 1
M = 4  # dow_sin, dow_cos, month_sin, month_cos

seq_x        = (112, 1)
seq_y        = (56, 1)
seq_x_mark   = (112, 4)
seq_y_mark   = (56, 4)

batch_x      = (B, 112, 1)
batch_y      = (B, 56, 1)
batch_x_mark = (B, 112, 4)
batch_y_mark = (B, 56, 4)
output       = (B, 56, 1)
```

시계열 한 구간은 이렇게 해석함:

```text
X_i = [cash_i, ..., cash_{i+111}]
Y_i = [cash_{i+112}, ..., cash_{i+167}]
```

의미:
- `cash_t`: 해당 날짜의 ATM cash withdrawal 값
- `dow_sin`, `dow_cos`: 요일 주기성을 원형 좌표로 표현
- `month_sin`, `month_cos`: 월별 계절성을 원형 좌표로 표현
- 현재 받은 파일은 `without_missing_values` 버전이라 missing이 이미 보정된 daily series로 취급
- 값은 train split 통계로 scaling되므로 예시 window에서는 음수/소수가 나올 수 있음

#### Car Parts

Car Parts는 2674개 car-part monthly sales series가 모두 같은 51개월 calendar grid를 공유함. 그래서 이번 구현에서는 각 part를 별도 sample로 나누지 않고, 한 시점의 판매 벡터를 `2674`차원 feature로 보는 true multivariate forecast 구조로 둠.

기본 예시:

```text
L = 24
P = 12
label_len = 12
D = 2674  # one channel per car-part series
M = 2     # month_sin, month_cos

seq_x        = (24, 2674)
seq_y        = (12, 2674)
seq_x_mark   = (24, 2)
seq_y_mark   = (12, 2)

batch_x      = (B, 24, 2674)
batch_y      = (B, 12, 2674)
batch_x_mark = (B, 24, 2)
batch_y_mark = (B, 12, 2)
output       = (B, 12, 2674)
```

시계열 한 구간은 이렇게 해석함:

```text
X_i = [v_i, ..., v_{i+23}]
v_t = [part_1_sales_t, part_2_sales_t, ..., part_2674_sales_t]

Y_i = [v_{i+24}, ..., v_{i+35}]
```

의미:
- `v_t`: 같은 월에 관측된 2674개 car part sales를 쌓은 벡터
- `D=2674`: 입력 feature dimension이자 예측 output channel dimension
- `month_sin`, `month_cos`: 월별 seasonality를 원형 좌표로 표현
- 길이가 51개월로 짧아서 window 기준 split을 사용함. 기본값이면 전체 16개 window 중 train/val/test가 대략 11/1/4로 나뉨
- 값은 train window 구간 통계로 channel별 scaling되므로 예시 window에서는 음수/소수가 나올 수 있음

#### Web Traffic

Web Traffic은 Wikipedia page 하나를 하나의 일별 traffic series로 봄. 일별 traffic은 요일/월 계절성이 강하므로 calendar mark를 실제 signal로 사용함.

기본 예시:

```text
L = 90
P = 30
label_len = 30
D = 1
M = 4  # dow_sin, dow_cos, month_sin, month_cos

seq_x        = (90, 1)
seq_y        = (30, 1)
seq_x_mark   = (90, 4)
seq_y_mark   = (30, 4)

batch_x      = (B, 90, 1)
batch_y      = (B, 30, 1)
batch_x_mark = (B, 90, 4)
batch_y_mark = (B, 30, 4)
output       = (B, 30, 1)
```

시계열 한 구간은 이렇게 해석함:

```text
X_i = [traffic_i, ..., traffic_{i+89}]
Y_i = [traffic_{i+90}, ..., traffic_{i+119}]
```

의미:
- `traffic_t`: 해당 날짜의 page view count
- `dow_sin`, `dow_cos`: 요일 주기성을 원형 좌표로 표현
- `month_sin`, `month_cos`: 월별 계절성을 원형 좌표로 표현
- missing value는 첫 구현에서 0으로 채우는 정책을 사용

<hr>

### dataset.py 요약 커맨드

사전 다운로드가 필요하면:

```bash
bash download.sh bespoke
bash download.sh generic
bash download.sh tourism_monthly nn5 car_parts web_traffic dominick_tsf
```

Dominick TSF 요약:

```bash
python dataset.py \
  --root_path ./dataset \
  --data_path dominick_dataset.tsf \
  --mode tsf \
  --seq_len 16 \
  --label_len 8 \
  --pred_len 8
```

Web Traffic 요약:

```bash
python dataset.py \
  --root_path ./dataset \
  --data_path kaggle_web_traffic_dataset_without_missing_values.tsf \
  --mode web \
  --seq_len 90 \
  --label_len 30 \
  --pred_len 30
```

Tourism Monthly 요약:

```bash
python dataset.py \
  --root_path ./dataset \
  --data_path tourism_monthly_dataset.tsf \
  --mode tourism \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24
```

NN5 Daily 요약:

```bash
python dataset.py \
  --root_path ./dataset \
  --data_path nn5_daily_dataset_without_missing_values.tsf \
  --mode nn5 \
  --seq_len 112 \
  --label_len 56 \
  --pred_len 56
```

Car Parts 요약:

```bash
python dataset.py \
  --root_path ./dataset \
  --data_path car_parts_dataset_without_missing_values.tsf \
  --mode car_parts \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12
```

Generic Monash TSF 요약 예시:

```bash
python dataset.py \
  --root_path ./dataset \
  --data_path cif_2016_dataset.tsf \
  --mode monash \
  --dataset_key cif_2016 \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 12
```

<hr>

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
