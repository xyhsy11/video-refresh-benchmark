# Prospective dataset decision after the HMDB population gate

The verified HMDB51 redesign does not pass the declared conservative full-class population gate. No dataset is acquired or switched in this commit, and no recognition outcomes are used to choose a replacement.

For a separate **input-only** proposal, the first candidate to investigate is **NTU RGB+D 60 with the cross-subject protocol**. The dataset authors describe 60 classes, 56,880 action samples, synchronized camera views, and explicit setup/camera/performer/repetition/action identifiers. These identifiers provide a better starting point for family accounting than HMDB title prefixes; they do not by themselves prove independence. Related camera views and repeated physical performances must remain together. See the [authors’ dataset description](https://github.com/shahroudy/NTURGB-D).

This candidate does **not** yet pass resource or access gates. The official page lists 136 GB of RGB videos, exceeding this study’s20 GiB network cap and current free workspace capacity. Registration, a request and a release agreement are required; the stated use is noncommercial academic research. No account, agreement acceptance or request was made here. A separately approved access/storage/compute plan, or preexisting authorized local bytes, is necessary. See the [official access, sizes and terms](https://rose1.ntu.edu.sg/dataset/actionRecognition/).

Before any model call, use the following fixed decision framework:

|Gate|Required evidence before considering GO|
|---|---|
|Source and role independence|Authenticate original identifiers; use cross-subject boundaries, group synchronized views/physical takes and interacting participants conservatively; audit duplicate links and pretraining overlap. Do not count camera views as independent sources.|
|All target classes|Declare all60 classes prospectively; verify actual per-role clip and family minima with global allocation. Do not choose classes using accuracy.|
|Real inputs and labels|Authorized original RGB bytes, checksums, annotations and a complete auditable inventory; public catalog counts are planning information.|
|Temporal suitability|Fresh input-only actual-PTS screening; report coverage and snippet time spans without recognition-based substitutions.|
|Compute and storage|Approve the genuine RGB acquisition/storage budget; project CPU/GPU cost using training-only real smoke inputs only after a separately reviewed population protocol.|
|Licensing|User-reviewed access and agreement eligibility; retain terms and provenance; do not redistribute raw video.|
|Outcome independence|Choose population, thresholds and roles before examining evaluation predictions; no significance-driven switch or cohort extension.|

The next concrete step is a separate access/resource and source-identity feasibility proposal for this candidate, not neural training. If that proposal fails, apply the same framework to another prospectively chosen dataset. NTU is not represented as available, affordable, pretraining-clean or already eligible.

If a future independently justified HMDB population protocol instead passes, the exact next experiment is a newly sealed51-class frozen-X3D study with a new affine head, equal-architecture H/H+cheap controllers, train/development-only selection, actual causal B4 test executions before privileged dense diagnostics, and independently measured complete-pipeline cost. The present A/B/C failure does not authorize that experiment.
