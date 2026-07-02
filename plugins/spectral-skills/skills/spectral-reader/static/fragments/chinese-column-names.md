# Chinese Column Names

Common Chinese roles:

- sample ID: `样本编号`, `编号`, `样品编号`
- sequence or metadata: `序号`, `备注`, `批次`, `日期`, `仪器`
- label: `类别`, `等级`, `品种`, `产地`
- target: `含量`, `浓度`, `指标`
- band axis: `波长`, `波数`, `波段`

Do not treat `编号` or `序号` as spectral variables without evidence. Preserve
Chinese column names in the internal read settings and Data Contract.

## Reading Semantics

This scene usually affects `sample_id`, `label`, `target`, `metadata`,
`spectral_columns`, and `task_hint`.

Require confirmation when role words could indicate either metadata or labels,
such as grade/category fields. Return `blocked` if translated role evidence
conflicts and no safe role assignment can be described.

