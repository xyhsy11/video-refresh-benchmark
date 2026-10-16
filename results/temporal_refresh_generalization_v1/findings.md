# temporal_refresh_generalization_v1 findings

Primary new-source X3D-XS B4 cache effect: **+0.009784 nats/video**, paired 95% CI [-0.012442, +0.032961],100 sources, five per class. Positive means CE(feature oracle)>CE(periodic), hence worse classification loss. This is inconclusive at this precision; it is not evidence of equivalence.

Actual starting revision: 0c5396f80adc9bdc047bf3809826fb680a75767b; tracked inputs match the cited design revision. All 400 training and 100 development source roles are retained. New evaluation has 100 distinct parents, five per class; 697 prior/reserved parent IDs and 690 encoded hashes excluded. 11 distinct technically ineligible candidates were rejected before freeze. New HTTP body downloads: 3.028294GiB; conservative interruption charge: 0 bytes;20 GiB cap. Reused compressed archives are accounted separately in data_plan.json.

## Frozen recognition heads

|Representation|Selected alpha|Train dense accuracy|Train CE|Development dense accuracy|Development CE|LBFGS iterations|Maximum objective gradient|
|---|---:|---:|---:|---:|---:|---:|---:|
|resnet18_endpoint|0.1|100.0%|0.265722|70.0%|1.135178|127|4.24e-07|
|resnet18_four_frame_mean|0.01|100.0%|0.043741|70.0%|1.110655|177|1.23e-07|
|x3d_xs|0.01|100.0%|0.007688|90.0%|0.444315|190|2.29e-07|

All grids and solver records are retained. Dense sanity checks are reported without filtering sources or requiring an arbitrary favorable accuracy.

## Quality, kept separate by cohort

|Cohort|Representation|Dense accuracy|Periodic B4 accuracy|Feature-oracle B4 accuracy|Periodic CE|Feature-oracle CE|d and95%CI|
|---|---|---:|---:|---:|---:|---:|---|
|development|resnet18_endpoint|70.0%|72.0%|70.0%|1.159642|1.121972|-0.037670 [-0.086996,+0.009997]|
|development|resnet18_four_frame_mean|70.0%|70.0%|70.0%|1.116943|1.134275|+0.017332 [-0.011183,+0.044715]|
|development|x3d_xs|90.0%|88.0%|88.0%|0.471060|0.478711|+0.007651 [-0.024987,+0.040344]|
|evaluation|resnet18_endpoint|74.0%|79.0%|76.0%|0.892882|0.875199|-0.017683 [-0.067185,+0.032561]|
|evaluation|resnet18_four_frame_mean|74.0%|73.0%|74.0%|0.793942|0.797422|+0.003480 [-0.019387,+0.025657]|
|evaluation|x3d_xs|87.0%|88.0%|86.0%|0.425848|0.435633|+0.009784 [-0.012442,+0.032961]|

Development is reused/tuned; evaluation uses new source IDs, frozen heads/settings/code and historical exclusions. It is not pooled with exploratory data. All 20 classes are the existing classes. Official training annotation parent overlap is excluded, but reuploads and universal checkpoint pretraining independence cannot be proved.

## Privileged task selection and pooling

|Cohort|Representation|H_C|H_U|L_C(S_C)|L_U(S_C)|L_C(S_U)|L_U(S_U)|
|---|---|---:|---:|---:|---:|---:|---:|
|development|resnet18_endpoint|0.464906|0.446555|0.694737|0.844085|0.824410|0.713087|
|development|resnet18_four_frame_mean|0.328805|0.377417|0.788138|0.907064|0.935700|0.739526|
|development|x3d_xs|0.236080|0.231334|0.234980|0.308048|0.307945|0.239725|
|evaluation|resnet18_endpoint|0.405864|0.404873|0.487018|0.619299|0.602899|0.488009|
|evaluation|resnet18_four_frame_mean|0.275423|0.334559|0.518519|0.588140|0.651504|0.459384|
|evaluation|x3d_xs|0.185008|0.179806|0.240840|0.292074|0.305883|0.246043|

SC/SU use true labels and every candidate feature. Cross-cells hold selected positions fixed; reoptimizing under the other pool changes both selection and weighting. Feature reconstruction error remains a last-cache-timeline metric. Positive task-oracle headroom is guaranteed by including periodic, not evidence a practical controller recovers it. CE minimization does not maximize top1.

## Predetermined budget sensitivities

|Representation|B|New-source d (nats)|95%CI|
|---|---:|---:|---|
|resnet18_endpoint|2|-0.060583|[-0.151904,+0.029780]|
|resnet18_endpoint|8|-0.005491|[-0.029584,+0.018568]|
|resnet18_four_frame_mean|2|+0.018094|[-0.054611,+0.088511]|
|resnet18_four_frame_mean|8|+0.007432|[-0.004550,+0.021147]|
|x3d_xs|2|+0.032261|[-0.015414,+0.080887]|
|x3d_xs|8|-0.003616|[-0.016401,+0.008744]|

These are secondary marginal/unadjusted comparisons. The primary representation,budget,pool and sign were fixed before evaluation.

## Measured system costs

|Representation|Dense16 mean per-video median ms|PeriodicB4 mean per-video median ms|
|---|---:|---:|
|resnet18_endpoint|817.081|643.170|
|resnet18_four_frame_mean|1486.705|797.782|
|x3d_xs|3187.198|1227.438|

The boundary starts with encoded bytes in memory and includes a fresh full sequential decode,within-trial frame buffering and timestamp candidate planning,RGB preparation,transfers,float32 encoders,float64 pooling/head and completed prediction. Disk loading/setup are separate. No predecoded input or free candidate inspection is used. C also executes its native 400-class projection before discarding those logits; that implementation cost is included. Two warmups and 5 measured trials/mode/source; 20 fixed evaluation sources, one per class. Raw decoded/call/image counters,trial variability,throughput and memory scopes are saved. A uses 16/4 frame calls; B uses 64/16 independent frame calls; C uses 16/4 four-frame calls. Equal refresh budgets are not equal compute. Memory reports torch allocated resident and per-trial incremental peaks,plus process working set; it is not whole-device physical or exact per-trial CPU peak memory. No oracle is assigned measured deployable latency.

## Limits and next decision

The 400/100 reused source roles replace the older 600-new-source proposal. Three new heads select regularization by lowest dense development CE (historically accuracy first), then accuracy/grid order. Training-only normalization and solver diagnostics are saved. Confidence intervals use 10,000 class-stratified whole-source resamples,PCG64 seed 20261024,linear percentiles,shared indices. They condition on fixed heads/classes and omit training/model-selection/repeated-research uncertainty. Development-derived precision was estimated before evaluation;100 sources did not guarantee a conclusive interval.

A/B/C use the same endpoints and four-frame support. B-versus-C does not isolate a pure temporal-model effect: architecture,pretraining,resolution,feature dimension,accuracy and calibration differ. Full-clip candidate construction is offline even with closed trailing snippets. This study says nothing definitive about whether earlier gain predictors work on X3D. These exposed evaluation sources cannot become untouched confirmation for a later controller. No adaptive speedup,streaming system,recognition guarantee or publication acceptance is established.

Next: Plan one precision-justified new-source replication of this fixed primary comparison before building another controller; the present interval does not resolve the proxy question. No follow-up acquisition or controller training is launched.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe .\run_temporal_refresh_generalization.py --device cuda --max-download-gib 20 --resume
& .\.venv_temporal\Scripts\python.exe -m unittest test_temporal_refresh_generalization -v
& .\.venv_temporal\Scripts\python.exe .\run_temporal_refresh_generalization.py --audit-only
```


## Executed work and hardware

The run retained 28,800 reference embeddings(600 videos ? 16 candidates ? 3 representations). Logged collection/smoke/verification starts were 58,023;successful warm-up/measured timing added 16,800 encoder invocations,for 74,823 initiated calls. Ninety starts came from interrupted pre-exposure collection attempts;their individual completion is not attested. Smoke/verification calls are separate from retained dataset collection. Twelve classifier-grid fits were executed;no backbone or refresh-gain predictor was trained.

Hardware: Intel Core i7-10870H (8 cores / 16 logical),NVIDIA GeForce RTX3080 Laptop GPU 16 GiB,driver 610.62. PyTorch 2.14.1+cu132 reports torch.version.cuda 13.2;this differs from the driver's advertised maximum CUDA compatibility. Python 3.12.10,torchvision 0.29.1+cu132,PyAV 19.0.1 and all linked FFmpeg versions are recorded in environment.json. Added dependencies live in .venv_temporal;the historical .venv was not upgraded. The user closed a competing GPU game before timing;device-context samples are retained.

Torch allocated resident memory was 94,299,648 bytes for both encoders/three heads,with 20,466,176?28,403,712 bytes additional peak allocation during trials. CPU working-set samples ranged 1,356,398,592?1,806,831,616 bytes;these are process samples,not isolated inference peaks. Per-model parameter/buffer sizes and model-loading setup time are in setup_sessions.json.

One pre-exposure archive retry followed a Windows concurrent-ledger write error; 332,398,592 partial body bytes are included in the 3.028294 GiB accounting. The public copy replaces only project-absolute error paths with relative paths;the original ledger is preserved in ignored local assets with its SHA256. No byte counters or scientific measurements were changed. See public_sanitization.json and metadata.json.
