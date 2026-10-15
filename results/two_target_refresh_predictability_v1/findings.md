# two_target_refresh_predictability_v1 findings

This fixed saved-table diagnostic uses exactly 100 development sources/900 opportunities and 40 exploratory sources/360 opportunities. All nine observations from a source share its original fold. Both cohorts are developmental, not fresh confirmations: recognition regularization previously used development sources, and exploratory sources have been inspected. No RGB encoder, video decoder, video acquisition or recognition training runs. New fixed probes are fitted; the recognition head and prior artifacts remain unchanged.

Primary inputs are the exact 56 C columns, never the privileged P columns. Targets B=LA-LB, C=LC-LB and G=LA-LC remain signed nats. Averaged attribution is one predeclared sensitivity analysis. The four schedules retain the old meanings and unequal-budget B/D controls. The labels are retrospective whole-video task outcomes, not online observations.

## Answers to the research questions

1. **No tested learned probe improved component MAE over its own training-median baseline.** Forward RF benefit MAE is 0.104609 versus 0.091360 on development, and 0.104759 versus 0.093164 on exploratory. Removal-cost MAE is 0.224828 versus 0.204342, and 0.249368 versus 0.233596. All these MAE-improvement intervals are negative. Ridge probes and averaged-attribution component probes also fail this baseline comparison. RF benefit AUROC is only descriptively 0.5203/0.5443, while cost AUROC is 0.5102/0.4885; these weak rankings do not establish useful task-value prediction.
2. **Ridge equivalence passes.** Maximum discrepancy is 4.5242e-15, with zero threshold decision disagreements. Previous C-linear, C-nonlinear and age/budget predictions reproduce exactly. Separate ridge targets provide no algorithmic advantage under these fixed linear-in-target estimators.
3. **The forest comparison is mixed.** Forward two-target minus direct-G RF400 utility is +0.013350 [+0.002634, +0.025406] nats per decision on development OOF videos, but -0.014044 [-0.046467, +0.014161] on exploratory videos. This relative advantage is not consistent across cohorts.
4. **Absolute positive utility is not established.** Forward two-target utility is -0.011775 [-0.030190, +0.006186] on development and -0.078571 [-0.222695, +0.007420] on exploratory. Both point estimates are negative and both intervals include zero. A less harmful result relative to direct-G RF400 does not make useful refresh decisions.
5. **Averaged attribution does not rescue the conclusion.** Its utilities are -0.014353 [-0.033520, +0.005320] and -0.088155 [-0.237818, +0.003859]. The averaged-minus-forward differences include zero in both cohorts. The relative development advantage over RF400 no longer has an entirely positive interval under averaged attribution, while neither order establishes positive absolute utility.
6. **Next: revisit the task and temporal representation with more independent-source support before another controller variant.** Freeze one diagnostic testing a richer legitimately cached temporal observation against constants and direct-G prediction on the same bounded target. Do not start new-source confirmation of a useful decomposition method yet: the predeclared absolute-utility requirement failed.

The component-error cross-products are negative for forward RF (-0.024288 development; -0.068912 exploratory), so they add rather than cancel in net MSE. The verified identity yields net MSE 0.230437 and 0.452609. This is an uncentered error-product statement, not a covariance-only or causal explanation.

## Decision quality

Each independent move is selected only when predicted G>1e-12. Utility is actual G times selection, averaged over all nine opportunities within source (including zeros), then over sources. Keeping reference is zero. No independent moves are combined into a policy.

|Cohort|Method|Utility and 95% interval (nats/decision)|Selected / total|Helpful / harmful / neutral|Mean selected gain (denominator)|
|---|---|---|---:|---|---|
|development_oof|C_linear_direct|-0.031328 [-0.054248, -0.010123]|401/900|171/230/0|-0.070311 (401)|
|development_oof|C_linear_two_target_averaged|-0.031328 [-0.054248, -0.010123]|401/900|171/230/0|-0.070311 (401)|
|development_oof|C_linear_two_target_forward|-0.031328 [-0.054248, -0.010123]|401/900|171/230/0|-0.070311 (401)|
|development_oof|C_nonlinear_direct|-0.014451 [-0.039741, +0.012648]|423/900|178/245/0|-0.030746 (423)|
|development_oof|C_nonlinear_two_target_averaged|-0.014451 [-0.039741, +0.012648]|423/900|178/245/0|-0.030746 (423)|
|development_oof|C_nonlinear_two_target_forward|-0.014451 [-0.039741, +0.012648]|423/900|178/245/0|-0.030746 (423)|
|development_oof|Direct_G_RF200|-0.025096 [-0.042472, -0.009068]|287/900|110/177/0|-0.078699 (287)|
|development_oof|Direct_G_RF400|-0.025125 [-0.042401, -0.009087]|285/900|107/178/0|-0.079343 (285)|
|development_oof|Two_target_RF200x2_averaged|-0.014353 [-0.033520, +0.005320]|331/900|142/189/0|-0.039025 (331)|
|development_oof|Two_target_RF200x2_forward|-0.011775 [-0.030190, +0.006186]|284/900|122/162/0|-0.037314 (284)|
|development_oof|age_budget_ridge|-0.002988 [-0.012985, +0.003364]|53/900|29/24/0|-0.050732 (53)|
|development_oof|keep_reference|+0.000000 [+0.000000, +0.000000]|0/900|0/0/0|undefined: no selections|
|development_oof|training_mean|+0.000000 [+0.000000, +0.000000]|0/900|0/0/0|undefined: no selections|
|development_oof|training_median|+0.000000 [+0.000000, +0.000000]|0/900|0/0/0|undefined: no selections|
|exploratory|C_linear_direct|-0.058499 [-0.146884, +0.005358]|150/360|62/88/0|-0.140397 (150)|
|exploratory|C_linear_two_target_averaged|-0.058499 [-0.146884, +0.005358]|150/360|62/88/0|-0.140397 (150)|
|exploratory|C_linear_two_target_forward|-0.058499 [-0.146884, +0.005358]|150/360|62/88/0|-0.140397 (150)|
|exploratory|C_nonlinear_direct|-0.088699 [-0.220018, -0.002820]|151/360|67/84/0|-0.211469 (151)|
|exploratory|C_nonlinear_two_target_averaged|-0.088699 [-0.220018, -0.002820]|151/360|67/84/0|-0.211469 (151)|
|exploratory|C_nonlinear_two_target_forward|-0.088699 [-0.220018, -0.002820]|151/360|67/84/0|-0.211469 (151)|
|exploratory|Direct_G_RF200|-0.067535 [-0.208755, +0.016997]|120/360|52/68/0|-0.202606 (120)|
|exploratory|Direct_G_RF400|-0.064527 [-0.202871, +0.016997]|121/360|54/67/0|-0.191980 (121)|
|exploratory|Two_target_RF200x2_averaged|-0.088155 [-0.237818, +0.003859]|123/360|50/73/0|-0.258015 (123)|
|exploratory|Two_target_RF200x2_forward|-0.078571 [-0.222695, +0.007420]|108/360|48/60/0|-0.261904 (108)|
|exploratory|age_budget_ridge|-0.000201 [-0.000635, +0.000031]|2/360|1/1/0|-0.036251 (2)|
|exploratory|keep_reference|+0.000000 [+0.000000, +0.000000]|0/360|0/0/0|undefined: no selections|
|exploratory|training_mean|+0.000000 [+0.000000, +0.000000]|0/360|0/0/0|undefined: no selections|
|exploratory|training_median|+0.000000 [+0.000000, +0.000000]|0/360|0/0/0|undefined: no selections|

## Predeclared comparisons

Positive paired differences favor the candidate. Primary is forward two-target RF200x2 minus direct-G RF400 on development OOF videos. Absolute utility must also be positive: a less negative result is not a useful policy. Secondary intervals are marginal and unadjusted.

|Cohort|Candidate minus baseline|Mean paired gain and 95% interval (nats)|Improved / tied / worsened videos|
|---|---|---|---|
|development_oof|Two_target_RF200x2_forward minus Direct_G_RF400|+0.013350 [+0.002634, +0.025406]|27/56/17|
|development_oof|Two_target_RF200x2_forward minus keep_reference|-0.011775 [-0.030190, +0.006186]|18/48/34|
|development_oof|Two_target_RF200x2_forward minus Direct_G_RF200|+0.013322 [+0.002466, +0.025421]|29/54/17|
|development_oof|Two_target_RF200x2_forward minus C_nonlinear_direct|+0.002676 [-0.019131, +0.022593]|36/30/34|
|development_oof|Two_target_RF200x2_forward minus C_linear_direct|+0.019553 [-0.001545, +0.042196]|35/29/36|
|development_oof|Two_target_RF200x2_averaged minus Two_target_RF200x2_forward|-0.002578 [-0.014449, +0.011292]|17/57/26|
|development_oof|Two_target_RF200x2_averaged minus Direct_G_RF400|+0.010773 [-0.003046, +0.026187]|33/43/24|
|exploratory|Two_target_RF200x2_forward minus Direct_G_RF400|-0.014044 [-0.046467, +0.014161]|11/17/12|
|exploratory|Two_target_RF200x2_forward minus keep_reference|-0.078571 [-0.222695, +0.007420]|9/17/14|
|exploratory|Two_target_RF200x2_forward minus Direct_G_RF200|-0.011036 [-0.045064, +0.020278]|11/17/12|
|exploratory|Two_target_RF200x2_forward minus C_nonlinear_direct|+0.010128 [-0.029102, +0.051276]|19/9/12|
|exploratory|Two_target_RF200x2_forward minus C_linear_direct|-0.020073 [-0.098347, +0.040530]|19/9/12|
|exploratory|Two_target_RF200x2_averaged minus Two_target_RF200x2_forward|-0.009584 [-0.033861, +0.007639]|8/23/9|
|exploratory|Two_target_RF200x2_averaged minus Direct_G_RF400|-0.023628 [-0.063522, +0.009524]|12/13/15|

Development primary relative utility: +0.013350 [+0.002634, +0.025406]. Absolute forward two-target utility: -0.011775 [-0.030190, +0.006186].
The predeclared joint requirements are not established. Do not claim successful decomposition-based decisions or that two targets are necessary.

## Component predictability

Compare each target with its own training-median predictor. Raw B-MAE versus raw C-MAE is not an easier/harder comparison because scales differ. All means give sources equal weight. Pooled Spearman and positive-target AUROC are descriptive; nine opportunities per video are correlated. Constant scores or single-class outcomes are null with reasons, not invented values.

|Cohort|Target|Recipe|MAE (nats)|Own median MAE|MAE improvement and 95% interval|MSE (nats squared)|Spearman|Positive-target AUROC|
|---|---|---|---:|---:|---|---:|---|---|
|development_oof|B|C_linear|0.134053|0.091360|-0.042693 [-0.051383, -0.034310]|0.041546|-0.09146276312275282|0.45750794301780323; events 439/461|
|development_oof|B_averaged|C_linear|0.282281|0.195779|-0.086503 [-0.104105, -0.069371]|0.174548|-0.09177073058114886|0.46338438735177867; events 440/460|
|development_oof|C|C_linear|0.277145|0.204342|-0.072802 [-0.088795, -0.057085]|0.183494|-0.02351698376582358|0.48883751782037066; events 472/428|
|development_oof|C_averaged|C_linear|0.265008|0.200975|-0.064033 [-0.079758, -0.048310]|0.160976|-0.009607501573047211|0.5145198592257416; events 510/390|
|development_oof|G|C_linear|0.348287|0.253471|-0.094817 [-0.115822, -0.074784]|0.293865|-0.0342095730982384|0.46995364475568996; events 412/488|
|development_oof|G|C_linear_two_target_averaged|0.348287|0.253471|-0.094817 [-0.115822, -0.074784]|0.293865|-0.0342095730982384|0.46995364475568996; events 412/488|
|development_oof|G|C_linear_two_target_forward|0.348287|0.253471|-0.094817 [-0.115822, -0.074784]|0.293865|-0.0342095730982384|0.46995364475568996; events 412/488|
|development_oof|B|C_nonlinear|0.114002|0.091360|-0.022642 [-0.029200, -0.016611]|0.032552|0.024135733500905556|0.5024236704401148; events 439/461|
|development_oof|B_averaged|C_nonlinear|0.241733|0.195779|-0.045954 [-0.059938, -0.032884]|0.138333|0.02057553157473034|0.5061758893280632; events 440/460|
|development_oof|C|C_nonlinear|0.244139|0.204342|-0.039797 [-0.049862, -0.030013]|0.153908|-0.011434713705407868|0.4818628227467131; events 472/428|
|development_oof|C_averaged|C_nonlinear|0.237035|0.200975|-0.036060 [-0.047482, -0.024699]|0.134802|0.07845668945270304|0.5451533433886375; events 510/390|
|development_oof|G|C_nonlinear|0.304876|0.253471|-0.051406 [-0.065260, -0.037879]|0.238098|-0.042329385591834065|0.4427025306382301; events 412/488|
|development_oof|G|C_nonlinear_two_target_averaged|0.304876|0.253471|-0.051406 [-0.065260, -0.037879]|0.238098|-0.042329385591834065|0.4427025306382301; events 412/488|
|development_oof|G|C_nonlinear_two_target_forward|0.304876|0.253471|-0.051406 [-0.065260, -0.037879]|0.238098|-0.042329385591834065|0.4427025306382301; events 412/488|
|development_oof|G|Direct_G_RF200|0.280198|0.253471|-0.026728 [-0.037635, -0.016323]|0.231721|-0.0425507895948867|0.44019079261499283; events 412/488|
|development_oof|G|Direct_G_RF400|0.280248|0.253471|-0.026778 [-0.037826, -0.016190]|0.231720|-0.0468156315474492|0.43724385245901637; events 412/488|
|development_oof|B_averaged|Two_target_RF200x2_averaged|0.224604|0.195779|-0.028826 [-0.040039, -0.018550]|0.135170|0.04453233119719613|0.5216180830039525; events 440/460|
|development_oof|C_averaged|Two_target_RF200x2_averaged|0.218615|0.200975|-0.017640 [-0.027516, -0.007905]|0.130639|0.10429363402078157|0.5630668677727502; events 510/390|
|development_oof|G|Two_target_RF200x2_averaged|0.287795|0.253471|-0.034325 [-0.047296, -0.021832]|0.231831|0.0015873768242012924|0.47129655419385647; events 412/488|
|development_oof|B|Two_target_RF200x2_forward|0.104609|0.091360|-0.013249 [-0.018432, -0.008482]|0.031504|0.049768991651137136|0.5202837250900538; events 439/461|
|development_oof|C|Two_target_RF200x2_forward|0.224828|0.204342|-0.020486 [-0.030066, -0.011216]|0.150357|0.028498769314929343|0.5102318628227467; events 472/428|
|development_oof|G|Two_target_RF200x2_forward|0.281674|0.253471|-0.028204 [-0.040390, -0.016694]|0.230437|-0.0010468079111803984|0.46713353493554033; events 412/488|
|development_oof|G|age_budget_ridge|0.264103|0.253471|-0.010633 [-0.015393, -0.005987]|0.219746|-0.03030488046944334|0.5159035293649531; events 412/488|
|development_oof|G|keep_reference|0.253456|0.253471|+0.000014 [-0.000137, +0.000165]|0.218374|undefined: constant input or outcome|undefined: constant score; events 412/488|
|development_oof|B|training_mean|0.093342|0.091360|-0.001983 [-0.002848, -0.001192]|0.030901|-0.10262461138692773|0.45819724378517535; events 439/461|
|development_oof|B_averaged|training_mean|0.203819|0.195779|-0.008040 [-0.011023, -0.005150]|0.131148|-0.08756780032800762|0.4653162055335968; events 440/460|
|development_oof|C|training_mean|0.215688|0.204342|-0.011345 [-0.016445, -0.006143]|0.141377|0.0024101784322071375|0.5204934262632662; events 472/428|
|development_oof|C_averaged|training_mean|0.216413|0.200975|-0.015438 [-0.022731, -0.008039]|0.123494|-0.04012931969308496|0.5108597285067873; events 510/390|
|development_oof|G|training_mean|0.261289|0.253471|-0.007818 [-0.012279, -0.003305]|0.218163|-0.004251832968234924|0.5188007321343308; events 412/488|
|development_oof|B|training_median|0.091360|0.091360|+0.000000 [+0.000000, +0.000000]|0.030755|-0.11675001095636255|0.4501924606802089; events 439/461|
|development_oof|B_averaged|training_median|0.195779|0.195779|+0.000000 [+0.000000, +0.000000]|0.131540|-0.07414095937568681|0.45642292490118574; events 440/460|
|development_oof|C|training_median|0.204342|0.204342|+0.000000 [+0.000000, +0.000000]|0.143022|-0.04680947547319232|0.45901314747346744; events 472/428|
|development_oof|C_averaged|training_median|0.200975|0.200975|+0.000000 [+0.000000, +0.000000]|0.127571|-0.0419135170268393|0.49638009049773757; events 510/390|
|development_oof|G|training_median|0.253471|0.253471|+0.000000 [+0.000000, +0.000000]|0.218284|-0.025946462921376708|0.4722465382778927; events 412/488|
|exploratory|B|C_linear|0.123767|0.093164|-0.030602 [-0.041254, -0.020942]|0.053203|0.04326010745967689|0.5172818532818533; events 175/185|
|exploratory|B_averaged|C_linear|0.262405|0.199013|-0.063392 [-0.086323, -0.041895]|0.217337|0.062122392919698455|0.5319996295267204; events 177/183|
|exploratory|C|C_linear|0.292677|0.233596|-0.059080 [-0.081688, -0.036128]|0.288599|-0.06904322821420947|0.4535052326314462; events 206/154|
|exploratory|C_averaged|C_linear|0.273254|0.206981|-0.066273 [-0.094160, -0.036843]|0.191184|-0.001174391777714334|0.4802116670363446; events 209/151|
|exploratory|G|C_linear|0.368692|0.295157|-0.073536 [-0.099006, -0.048666]|0.492374|-0.11293759982715915|0.4351279788172992; events 154/206|
|exploratory|G|C_linear_two_target_averaged|0.368692|0.295157|-0.073536 [-0.099006, -0.048666]|0.492374|-0.11293759982715915|0.4351279788172992; events 154/206|
|exploratory|G|C_linear_two_target_forward|0.368692|0.295157|-0.073536 [-0.099006, -0.048666]|0.492374|-0.11293759982715915|0.4351279788172992; events 154/206|
|exploratory|B|C_nonlinear|0.117739|0.093164|-0.024575 [-0.033102, -0.016213]|0.048111|-0.04645251892375713|0.47623166023166025; events 175/185|
|exploratory|B_averaged|C_nonlinear|0.249747|0.199013|-0.050734 [-0.070250, -0.031395]|0.197815|-0.01985740630714743|0.4981939427618783; events 177/183|
|exploratory|C|C_nonlinear|0.272325|0.233596|-0.038728 [-0.053070, -0.024532]|0.292752|-0.06489556246575975|0.46954986760812; events 206/154|
|exploratory|C_averaged|C_nonlinear|0.253573|0.206981|-0.046591 [-0.065872, -0.028681]|0.194447|-0.07303245652615632|0.4506796793307773; events 209/151|
|exploratory|G|C_nonlinear|0.339828|0.295157|-0.044671 [-0.061926, -0.027975]|0.478741|-0.05818537694478095|0.4750346740638003; events 154/206|
|exploratory|G|C_nonlinear_two_target_averaged|0.339828|0.295157|-0.044671 [-0.061926, -0.027975]|0.478741|-0.05818537694478095|0.4750346740638003; events 154/206|
|exploratory|G|C_nonlinear_two_target_forward|0.339828|0.295157|-0.044671 [-0.061926, -0.027975]|0.478741|-0.05818537694478095|0.4750346740638003; events 154/206|
|exploratory|G|Direct_G_RF200|0.309887|0.295157|-0.014730 [-0.029085, -0.000949]|0.457136|-0.0197275616027019|0.4980771655528937; events 154/206|
|exploratory|G|Direct_G_RF400|0.309259|0.295157|-0.014102 [-0.029064, +0.000032]|0.460252|-0.019423517520632122|0.4976673811625268; events 154/206|
|exploratory|B_averaged|Two_target_RF200x2_averaged|0.219085|0.199013|-0.020072 [-0.036011, -0.006478]|0.197492|0.045671109089497994|0.5557253558087123; events 177/183|
|exploratory|C_averaged|Two_target_RF200x2_averaged|0.238637|0.206981|-0.031656 [-0.046759, -0.017223]|0.189467|-0.10375107817912133|0.4278018948635888; events 209/151|
|exploratory|G|Two_target_RF200x2_averaged|0.327427|0.295157|-0.032271 [-0.051040, -0.014918]|0.480317|-0.08651030503795062|0.4709368301601311; events 154/206|
|exploratory|B|Two_target_RF200x2_forward|0.104759|0.093164|-0.011595 [-0.019091, -0.004762]|0.047745|0.058549898491296125|0.5442625482625483; events 175/185|
|exploratory|C|Two_target_RF200x2_forward|0.249368|0.233596|-0.015772 [-0.026591, -0.005172]|0.267039|-0.002877731148780022|0.4884945151935443; events 206/154|
|exploratory|G|Two_target_RF200x2_forward|0.315446|0.295157|-0.020290 [-0.034425, -0.005848]|0.452609|-0.048264411624251054|0.48676081200353044; events 154/206|
|exploratory|G|age_budget_ridge|0.303550|0.295157|-0.008393 [-0.016160, -0.000965]|0.425915|-0.026376367341456076|0.4866504854368932; events 154/206|
|exploratory|G|keep_reference|0.295243|0.295157|-0.000087 [-0.000277, +0.000094]|0.430683|undefined: constant input or outcome|undefined: constant score; events 154/206|
|exploratory|B|training_mean|0.094945|0.093164|-0.001781 [-0.002868, -0.000691]|0.042370|undefined: constant input or outcome|undefined: constant score; events 175/185|
|exploratory|B_averaged|training_mean|0.208059|0.199013|-0.009046 [-0.013519, -0.004490]|0.170439|undefined: constant input or outcome|undefined: constant score; events 177/183|
|exploratory|C|training_mean|0.243411|0.233596|-0.009815 [-0.017259, -0.002292]|0.260754|undefined: constant input or outcome|undefined: constant score; events 206/154|
|exploratory|C_averaged|training_mean|0.229716|0.206981|-0.022735 [-0.033150, -0.012104]|0.174680|undefined: constant input or outcome|undefined: constant score; events 209/151|
|exploratory|G|training_mean|0.301426|0.295157|-0.006269 [-0.012853, +0.000537]|0.423503|undefined: constant input or outcome|undefined: constant score; events 154/206|
|exploratory|B|training_median|0.093164|0.093164|+0.000000 [+0.000000, +0.000000]|0.042059|undefined: constant input or outcome|undefined: constant score; events 175/185|
|exploratory|B_averaged|training_median|0.199013|0.199013|+0.000000 [+0.000000, +0.000000]|0.168183|undefined: constant input or outcome|undefined: constant score; events 177/183|
|exploratory|C|training_median|0.233596|0.233596|+0.000000 [+0.000000, +0.000000]|0.266927|undefined: constant input or outcome|undefined: constant score; events 206/154|
|exploratory|C_averaged|training_median|0.206981|0.206981|+0.000000 [+0.000000, +0.000000]|0.180959|undefined: constant input or outcome|undefined: constant score; events 209/151|
|exploratory|G|training_median|0.295157|0.295157|+0.000000 [+0.000000, +0.000000]|0.430424|undefined: constant input or outcome|undefined: constant score; events 154/206|

For forward B/C, the following comparisons have an entirely positive MAE-improvement interval over their own training median (absence is not proof of no possible signal):
- development_oof: none of the fixed probes establishes a positive interval.
- exploratory: none of the fixed probes establishes a positive interval.

## Equivalence, prediction errors and resource accounting

Linear/kernel identities pass for both attributions and cohorts. Maximum heldout direct-G versus B-minus-C discrepancy is 4.52415882535e-15; previous direct predictions reproduce within 0. These recipes are linear in training targets, including the input-nonlinear RBF estimator; component heads do not introduce another decision method mathematically.
Threshold decision disagreements across equivalence controls: 0. Any disagreements must lie within the documented numerical equivalence band; they cannot support an algorithmic gain.

Separate target-dependent forest partitions are confirmed. Equal total tree count does not guarantee equal node count, capacity, serialized bytes or latency. Compressed joblib sizes use protocol5/compress3 and include estimator state, not a latency measurement.

|Final-development forest|Target|Trees|Nodes|Serialized bytes|
|---|---|---:|---:|---:|
|Direct_G_RF200|G|200|3198|87682|
|Direct_G_RF400|G|400|6244|169418|
|Two_target_RF200x2_forward|B|200|3208|86858|
|Two_target_RF200x2_forward|C|200|3518|95654|
|Two_target_RF200x2_averaged|B_averaged|200|3018|81641|
|Two_target_RF200x2_averaged|C_averaged|200|3718|100382|

The signed net error is eB-eC. Its MSE includes the **uncentered** error cross-product; correlated component errors can cancel or reinforce. This is not a covariance-only decomposition.

|Cohort|Method|MSE B|MSE C|Mean eB*eC|MSE G|Identity residual|
|---|---|---:|---:|---:|---:|---:|
|development_oof|C_linear_two_target_averaged|0.174548|0.160976|0.020830|0.293865|0|
|development_oof|C_linear_two_target_forward|0.041546|0.183494|-0.034412|0.293865|0|
|development_oof|C_nonlinear_two_target_averaged|0.138333|0.134802|0.017518|0.238098|0|
|development_oof|C_nonlinear_two_target_forward|0.032552|0.153908|-0.025819|0.238098|2.78e-17|
|development_oof|Two_target_RF200x2_averaged|0.135170|0.130639|0.016989|0.231831|5.55e-17|
|development_oof|Two_target_RF200x2_forward|0.031504|0.150357|-0.024288|0.230437|0|
|exploratory|C_linear_two_target_averaged|0.217337|0.191184|-0.041926|0.492374|0|
|exploratory|C_linear_two_target_forward|0.053203|0.288599|-0.075286|0.492374|5.55e-17|
|exploratory|C_nonlinear_two_target_averaged|0.197815|0.194447|-0.043239|0.478741|1.11e-16|
|exploratory|C_nonlinear_two_target_forward|0.048111|0.292752|-0.068938|0.478741|0|
|exploratory|Two_target_RF200x2_averaged|0.197492|0.189467|-0.046679|0.480317|5.55e-17|
|exploratory|Two_target_RF200x2_forward|0.047745|0.267039|-0.068912|0.452609|0|

## Interpretation and next experiment

Forward is primary; the averaged-attribution comparisons above are sensitivity results and never replace a losing primary method. Evaluate whether either attribution meets both relative and absolute requirements; similar component accuracy alone is insufficient. A difference between orders reflects nonlinear whole-video attribution, not a unique causal decomposition.
Unknown true labels, future-content dependence, limited source support and the cached representation remain unresolved. This fixed study cannot rule out every predictor. Full-clip candidates and future-reference codec vectors prevent a verified streaming-causal interpretation. End-to-end latency and adaptive recognition accuracy are unmeasured; no adaptive speedup or publication-level novelty follows.

Recommended single next experiment: Before another controller variant, freeze a task/temporal-representation diagnostic with greater independent-source support, testing whether richer legitimately cached temporal observations predict this same target beyond constant and direct-G baselines.

Verification passes: 102 model/scaler checks, 8400 bootstrap-root checks, unchanged folds/targets/prior artifacts, saved prediction/metric reproduction and shared whole-source bootstrap reproduction. Actual RGB encoder executions, video decodes, video downloads and recognition training calls are zero. Optional scikit-learn dependencies were installed separately without changing existing package versions.

Intervals use 10000 PCG64 whole-source resamples, seed 20261020, linear percentile convention. They condition on saved fitted probes and omit full training/model-selection/repeated-research uncertainty. Cohorts are separate and both previously inspected.

## Reproduce from the repository root

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\requirements-two-target.txt
& .\.venv\Scripts\python.exe -m unittest test_two_target_refresh_predictability -v
& .\.venv\Scripts\python.exe .\analyze_two_target_refresh_predictability.py --decomposition-input results/refresh_target_decomposition_v1 --ablation-input results/task_signal_ablation_v2 --output results/two_target_refresh_predictability_v1 --seed 20261020 --bootstrap 10000
& .\.venv\Scripts\python.exe .\analyze_two_target_refresh_predictability.py --decomposition-input results/refresh_target_decomposition_v1 --ablation-input results/task_signal_ablation_v2 --output results/two_target_refresh_predictability_v1 --seed 20261020 --bootstrap 10000 --audit-only
```

A completed invocation audits without overwriting. Source/input identity mismatches block reuse; choose a clearly versioned fresh directory for a changed protocol. No old scientific artifact is modified.
