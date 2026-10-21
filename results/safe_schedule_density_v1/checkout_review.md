# Checkout and preservation review

- Actual starting HEAD: `920cb698d814017d0e79a96ef5512de8c8cc7d5e`, on the
  existing `experiment/refresh-failure-evidence-v1` branch.
- Requested reviewed main: `6a7c433b8ac9df4ca4fadf12b46d0bf30fcdae9d`.
  Refreshing the Git remote metadata located this merge commit; its content tree
  is identical to the starting checkout. No later commits existed on origin/main
  at the protocol freeze. The full commands/results are represented by the
  actual/reviewed/origin fields in `preflight.json`; no reset was performed.
- Work continued on `codex/safe-schedule-density-v1` from that actual checkout.
  Initial unrelated `seminar_build/` and `seminar_output/` directories were left
  untouched and are excluded from the diagnostic commit.
- The only existing tracked file edited for this diagnostic is `.gitignore`,
  to retain the new result directory. Historical scientific files and historical
  workflow code are unchanged. The historical-result identity audit covers
  3,765 tracked result files using Git blob hashes without parsing protected
  scientific outcomes.
- All numerical work used local archived inputs. No dataset/model/package was
  downloaded; no model/head/scaler/controller fit, GPU initialization, encoder
  inference, or video decode occurred. Historical run/prepare commands were not
  invoked. The separate verifier needed two pandas-compatibility corrections
  before its successful read-only audit; sealed analyzer results were unchanged.
