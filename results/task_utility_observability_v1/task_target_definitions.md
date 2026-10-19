# Retrospective task targets and exact enumeration

EXPLORATORY, exposed development data. All f_t are genuine saved X3D-XS preprojection outputs. Frozen affine head h(f)=W((f-mu)/sigma)+b; unchanged original identities, K=101. CE(z,y)=logsumexp(z)-z_y in float64. Labels only construct retrospective targets; no label/current uncomputed feature/future embedding/teacher action enters a probe observation.

For t>0 let H be the EXACT recorded earlier refresh positions, including0, and r=max(H). Target A: gain_local=CE(h(f_r),y)-CE(h(f_t),y). Positive favors the current feature's immediate loss, not final video pooling. All signs, zero and near-zero gains are retained.

Enumerate S={(0,a,b,c):1<=a<b<c<=15}, |S|=455. A schedule is compatible iff its positions before t equal H, so earlier holds are also respected. Branch a in {hold,refresh} fixes whether t belongs to S. Exactly four refreshes are compulsory. Cache index c_s(u)=max{j in S:j<=u}; C pooled feature is (1/16) sum_u f_c_s(u). Affineness implies z_s=(1/16) sum_u h(f_c_s(u)); weights are reuse counts/16, not elapsed seconds.

Q_hold=min_s CE(z_s,y) over compatible schedules without t; Q_refresh=min_s CE(z_s,y) with t. Target B: gain_budget=Q_hold-Q_refresh. Positive favors refresh under perfect hindsight and globally optimal future continuation. Use literal float64 minimum values, preserve first lexicographic argmin for replay. Infeasible branches are null, not invented large numbers. Both branches are feasible exactly at discretionary states; forced/exhausted states remain documented but are excluded from predictability. Minimum continuation uses future genuine dense features and labels and is explicitly privileged, not an actually executed causal policy or treatment-effect estimate.

The proxy Z_pool teacher separately minimizes mean_k(centered(z_s)-centered(z_dense))^2 with the same compatible prefix; choose first objective within1e-12 of the minimum, preserving the historical rule. Its next action is a comparison target, never a causal feature or task-oracle substitute.

Independent replay uses combinations of remaining future positions separately for each branch and scalar actual-cache pooling/logaddexp CE, then a raw-feature affine pooling replay on hash-selected clips. This checks branch feasibility, sign, objective and pooling independently of the vectorized mask/matrix calculation. No new X3D extraction is performed.
