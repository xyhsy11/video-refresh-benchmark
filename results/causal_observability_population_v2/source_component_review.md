# Input-only review of conservative source components

This review uses filenames, official flags, byte hashes and saved pHash links only. It makes no visual recognition judgment and changes no relationship or exclusion. Every observed component is retained intact. All original parent URLs are missing, so independence of source-title components is uncertain.

|Representative title(s)|Clips|Classes|Official train / test / unused|Eligible train excluded|
|---|---:|---:|---:|---:|
|TheBoondockSaints|125|18|38 / 21 / 66|38|
|THE_PROTECTOR|100|10|76 / 0 / 24|0|
|Faith_Rewarded|91|8|23 / 28 / 40|23|
|AmericanGangster|74|9|25 / 10 / 39|25|
|KUNG_FU_HUSTLE|59|14|36 / 1 / 22|36|
|TheLastManOnearth|59|8|22 / 2 / 35|22|
|RETURN_OF_THE_KING|56|13|20 / 14 / 22|20|
|Veoh_Alpha_Dog_1|56|14|23 / 5 / 28|23|
|likebeckam|45|8|11 / 9 / 25|11|
|American_History_X|42|18|20 / 11 / 11|20|
|RATRACE|41|11|22 / 7 / 12|22|
|YouTube_smiles!|41|1|41 / 0 / 0|0|
|SweeneyTodd|40|9|13 / 10 / 17|13|
|Glory|39|15|12 / 14 / 13|12|
|HP_PRISONER_OF_AZKABAN|39|10|15 / 5 / 19|15|

There are 48 components with at least20 clips. Every component?s members, edge evidence and class-specific losses are recoverable from source_family_audit.json plus per_clip_eligibility.csv. No large-component split is justified by metadata alone.

The largest title components include film-level names such as TheBoondockSaints and THE_PROTECTOR. A film title can conservatively link multiple scenes while also obscuring different releases or sources; it does not prove that every clip is the same recording. Broad grouping can therefore reduce training availability beyond exact duplication. Relaxing it would require new independently verified parent-source provenance, not a desire to meet quotas. The all-training THE_PROTECTOR component still cannot cross the three training roles, so global assignment matters even without an official-test conflict.

The pHash criterion yields111 matching pairs, many already within shared-title components. This is an all-pairs edge audit, whereas the historical union procedure skipped pairs already connected. An increased reported edge count does not by itself imply111 new independent duplicate sources. Exact byte links and the cross-title Best_Kisses / Hangingbyamoment visual match are evidence for retaining connections even if titles differ.

|Grouping sensitivity|Available eligible train|Eligible test|Components|
|---|---:|---:|---:|
|exact_only|3568|1530|6761|
|exact_near|3568|1530|6676|
|title_only|2757|1530|1388|
|conservative_union_PRIMARY|2754|1530|1385|

Only the conservative union is primary. Less conservative counts remove operational protections and cannot establish GO. They show the effect of grouping assumptions; no recognition outcome was consulted.

For the weakest grid C, drink and eat have fewer available training clips than the33 total minimum; fall_floor, kick and kiss also lack the9 distinct components necessary to place at least3 in each disjoint training role. These are independently checkable necessary-condition failures; neither another MILP seed nor longer optimization can fix them under the same population rules.
