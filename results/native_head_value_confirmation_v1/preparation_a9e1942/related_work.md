# Comparability with primary research

This table positions a diagnostic; it is not a reproduced accuracy/latency league
table. Original papers use different datasets, trained systems and cost boundaries.

|Primary paper|Observations/lookahead|Backbone/head|Action space and aggregation|Objective|Cost comparability|
|---|---|---|---|---|---|
|[LiteEval, NeurIPS2019](https://papers.nips.cc/paper/8993-liteeval-a-coarse-to-fine-framework-for-resource-efficient-video-recognition)|Incoming coarse visual features; online/offline settings|Light and heavy CNNs, cooperative coarse/fine LSTMs|Gate expensive frame processing; recurrent recognition state|Joint recognition and resource allocation|Coarse acquisition/gating belongs to its system; E27 must measure its grayscale/HOG acquisition rather than assume it free|
|[AdaFrame, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html)|Frame history plus global memory/context and adaptive lookahead|CNN features and LSTM-based recognition/selection|Choose subsequent frame and stopping; recurrent prediction|Policy gradient and predicted future utility|Learned sequential frame use differs from12 independently reset interventions and fixed mandatory-first exact4|
|[AR-Net, ECCV2020](https://arxiv.org/abs/2007.15796)|Input-conditioned resolution policy over video frames|Jointly trained policy and recognition networks|Choose input resolution; video-level frame aggregation|Accuracy/efficiency tradeoff with differentiable training|Resolution-dependent recognition computation differs from fixed182 crop/X3D snippets; FLOPs alone do not validate decode/acquisition latency|
|[Uni-AdaFocus](https://arxiv.org/html/2412.11228v1)|Lightweight full-sequence features used for spatial/temporal allocation|Trainable light/heavy pathways; compatible with TSM/X3D|Spatial patches, dynamic frame sampling, conditional exit|End-to-end task supervision and dynamic computation|Its temporal/global information and trainable recognizer differ from a fixed native400 diagnostic; published hardware speed differs from this laptop timing boundary|
|[CoViAR, CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Compressed_Video_Action_CVPR_2018_paper.pdf)|I-frame images, compressed motion vectors and residuals|Separately trained compressed-representation streams|Temporal segment scores averaged before softmax; stream fusion|Action recognition using codec information|Codec representations have different dependencies/acquisition costs from decoded endpoint grayscale/HOG; no compressed-domain timing is measured in E27|
|Experiment27|Offline known full-clip grid, causal decision features, explicit decoder dependencies|Frozen native400 X3D-XS and fixed ridge probes|Exact4/16 refreshes, C logits pooling,12 independently reset alternatives|Signed CE benefit and separate cosine-drift prediction|Fresh complete-system timing controls; dense extraction/offline enumeration reported separately; no sequential adaptive speedup claim|

[Ross and Bagnell, Section2.5](https://arxiv.org/html/1406.5979v1#S2.SS5)
discuss the limitation that expert cost-to-go can be optimistic when the learner
cannot execute a comparably capable continuation. E25/E27's fixed-tail accounting
is a concrete diagnostic of continuation-dependent value, not discovery of that
general limitation or an implementation of AggreVaTe. This study does not train
the interacting policies studied in that paper. A later superior-adaptive-method
claim needs separate sequential execution and suitable learned baselines.

[TMLR's acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) do not
make this study's .01-nat/.005-nat references journal requirements. No acceptance
or publication-readiness percentage is inferred from software validation.
