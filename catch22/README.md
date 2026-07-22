# catch22 Feature Workspace

This folder is for sampling, managing, storing, and building data features for
symbolic proxy generation experiments.

## Valid-split catch24 sample features

Generate one random validation sample per forecasting dataset and pool
channel-wise catch24 features into one 24-dimensional feature vector per
dataset:

```bash
/home/gpuadmin/miniconda3/envs/tslib_nightly/bin/python catch22/extract_valid_sample_catch24.py
```

Default output:

```text
catch22/valid_sample_catch24_features.csv
```

By default, samples are loaded through the Time-Series-Library dataset classes
with train-split scaling enabled, matching the normal forecasting pipeline. Use
`--no-scale` to extract features from raw values.

## Valid-split sample correlation/distance analysis

Sample 10 validation sequences per dataset, extract channel-pooled catch22
features, summarize within-dataset variation, average samples into one
representative feature vector per dataset, and visualize between-dataset
distances:

```bash
/home/gpuadmin/miniconda3/envs/tslib_nightly/bin/python catch22/analyze_correlation_valid_sample.py
```

Default output directory:

```text
catch22/analysis_valid_samples/22 features/
```

## 53-dataset train-window PCA cluster circles

Sample 20 train windows from each of the 53 datasets, extract a channel-pooled
22-dimensional catch22 vector for every window, fit one global 2D PCA, and draw
only the per-dataset cluster circles:

```bash
/home/gpuadmin/miniconda3/envs/tslib_nightly/bin/python \
  catch22/plot_53_train_window_cluster_circles.py
```

The input length is 96 for every dataset except illness, which uses 36.  The
default circle center is the mean of the 20 2D PCA coordinates and the radius
is their maximum 2D distance from the center.  Sample coordinates are exported for
reproducibility but are not drawn in the plot.

Default output directory:

```text
catch22/train_window_cluster_circles_53/
```

## Global z-scored PCA with 53 dataset cluster circles

Sample 20 train windows from each dataset, globally z-score all 22 catch22
features, fit one shared 2D PCA, and plot all 1,060 samples together with 53
max-radius dataset circles:

```bash
/home/gpuadmin/miniconda3/envs/tslib_nightly/bin/python \
  catch22/visualize_53_train_window_pca_groups.py
```

Traffic and Weather are included in the six Benchmark datasets. The default
output directory is:

```text
catch22/train_window_pca_clusters_53/
```

## PCA-90 centroid clustering for 47 non-Benchmark datasets

Exclude ECL, Weather, Illness, Traffic, Exchange, and ETTh1; sample 20 train
windows from each of the remaining 47 datasets; globally z-score the 22
catch22 features; and retain the minimum PCA dimension explaining at least
90% of the variance. The script clusters the dataset centroids in that
retained PCA space into eight groups using Euclidean distance and
average-linkage agglomerative clustering:

```bash
/home/gpuadmin/miniconda3/envs/tslib_nightly/bin/python \
  catch22/cluster_47_dataset_centroids_pca90.py
```

No visualization is produced. The primary membership output and the compact
one-row-per-cluster summary are:

```text
catch22/dataset_centroid_clusters_47_pca90_k8/dataset_clusters_k8.csv
catch22/dataset_centroid_clusters_47_pca90_k8/cluster_summary_k8.csv
```
