# Leakage, provenance and evaluation-design audit

## Verified implementation checks

Original source-video IDs, not clip filenames, define folds. All nine opportunities stay together. Every held-out forest records only its fold-training source/example IDs; training-only population means, standard deviations, constant-column handling and720/900 training rows are independently reconstructed. Saved tree bootstrap root targets are checked. The motion rule receives only availability and p90 values from training rows to construct its median; it never optimizes against G. Final exploratory thresholds use development only. All frozen forest hyperparameters and >1e-12 decisions are retained. No model or threshold is refitted to improve audit outcomes.

Direct, benefit-minus-cost and history predictions are reproduced with explicit float32 tree traversal independent of original prediction/report functions; library predictions and archived selections are cross-checked. B=LA-LB,C=LC-LB,G=LA-LC=B-C are checked against saved arrays using separate explicit cache replay and stable cross entropy. Predictor matrices use whitelisted56 or100 columns; target/class/source metadata are excluded. No verified coding leakage was found in these checked paths. This is not proof against every possible defect or all upstream provenance errors.

## Permitted, inspected development data and design limitations

The frozen recognition head is fitted on400 source-disjoint training videos. Its normalization is independently reproduced from those dense mean-pooled references. Regularization0.01 was selected using all100 development videos (dense accuracy, then CE, then grid order). Therefore gain-predictor OOF is source-held-out only for that predictor; it is not a nested, fully independent head-plus-predictor evaluation. Recognition selection dependence is a design limitation, not an accidental coding bug.

The40 exploratory videos were excluded from head/probe fitting and selection in these frozen runs, but their outcomes had already been inspected in earlier research. Repeated representation/model choices across studies use reused cohorts. Saved per-run protocols can show fixed recipes before that run; they cannot prove decisions were uninformed by all previous outcomes or unrecorded human choices. Fixed seeds and no within-run tuning do not remove repeated-research selection uncertainty.

## Identity and unresolved provenance

All540 recognition-manifest records have distinct original source IDs and encoded SHA256 values. Source IDs agree with clip source prefixes; no same-parent windows cross cohorts, so no adjacent windows from a recorded parent leak across these cohorts. The100/40 audit memberships and original class-stratified five folds match saved identities. All540 saved reference-array hashes and candidate order/timestamps are independently checked. Encoded-file hashes are taken from the previously validated manifest; this audit does not rehash or decode video files.

Different source IDs/encoded hashes do not rule out reuploads, compilations, semantically near-duplicate content or a provenance error. No perceptual duplicate search or new raw-video inspection is conducted. Original Kinetics labels are reused; label correctness is not independently re-annotated. The data are a custom subset from CVDF validation archives, not an official full recognition benchmark. These unresolved matters are reported rather than assumed absent.

## Availability and offline scope

See input_availability.csv. Cached logits/history are legitimate pre-refresh observations under the unmodified periodic history, and repeated outputs add no new event. Candidate positions include full-clip first/last uniform selection. Codec vectors include future references in B-frames; restricting descriptor aggregation to a decoded prefix does not establish strict streaming causality. Privileged current RGB columns, true-label scores, future dense features and retrospective loss targets are not predictor inputs. Future references and labels are confined to the evaluator/target path. No nine independent moves are combined into a sequential schedule.

## Confirmation status

No genuinely independent test was run. Both current cohorts have been inspected. The completed protected100-video feature-fidelity confirmation is not repurposed, and its outcomes, features and motion tables are not read. No eligible untouched saved cohort is established. A prospective protocol is supplied without acquisition or evaluation.
