# Spectral Skills

闈㈠悜 Codex銆丆laude 绛夋櫤鑳戒綋鐨勭鍒扮鍏夎氨鏁版嵁鍒嗘瀽鎶€鑳介泦鍚堛€傞」鐩鐩栧厜璋辫鍙栥€佽川閲忔帶鍒躲€佹暟鎹垝鍒嗐€侀澶勭悊銆佺壒寰佸伐绋嬨€佸垎绫?鍥炲綊寤烘ā銆佸彈棰勭畻绾︽潫鐨勮嚜鍔ㄤ紭鍖栵紝浠ュ強璁烘枃绾у浘琛ㄤ笌鎶ュ憡杈撳嚭銆?
Spectral Skills is a leakage-aware agent skill collection for end-to-end spectral analysis. It ships Codex and Claude-compatible plugin metadata from one source tree.

## 鏍稿績鐗圭偣

- **绔埌绔絾涓嶅仛鎴愰粦绠?*锛氫節涓?skill 鍚勮嚜璐熻矗涓€涓竻鏅伴樁娈碉紝鐢?`spectral-workflow` 璺敱缁勫悎銆?- **闃叉鏁版嵁娉勬紡**锛氬厛鍒掑垎锛屽啀鎷熷悎棰勫鐞嗐€佺壒寰佸拰妯″瀷锛涙祴璇曢泦涓嶅弬涓庢柟娉曢€夋嫨鎴栬皟鍙傘€?- **瀹屾暣鏂规硶鑿滃崟**锛氭帹鑽愭柟妗堝彧鏄绠楀唴榛樿鍊硷紝涓嶄唬琛ㄧ郴缁熷彧鏀寔鎺ㄨ崘鍒楄〃涓殑鏂规硶銆?- **鑷姩璋冨弬鍙璁?*锛氬€欓€夌┖闂淬€乼rial銆侀€夋嫨鎸囨爣銆佹渶浣冲弬鏁板拰鏈€缁堢绾垮潎鍐欏叆鍚堝悓鏂囦欢銆?- **閫傞厤灏忔牱鏈厜璋?*锛氬悓鏃舵彁渚涘寲瀛﹁閲忓銆佷紶缁熸満鍣ㄥ涔犮€佸彲閫?boosting 鍜岄渶纭鐨勫疄楠屾繁搴︽ā鍨嬨€?- **缁撴灉鍙拷婧?*锛氶樁娈甸棿閫氳繃 JSON contract 鍜屾爣鍑?CSV 浜ゆ帴锛岃€屼笉鏄緷闈犱复鏃跺唴瀛樼姸鎬併€?- **璁烘枃绾х粯鍥?*锛氳緭鍑哄彲缂栬緫 SVG/PDF銆丳NG 棰勮銆佹簮鏁版嵁銆佸浘娉ㄣ€佺粯鍥句唬鐮佸拰 QA 璁板綍銆?
## 宸ヤ綔娴佹€昏

```text
鍘熷鍏夎氨鏂囦欢鎴栨枃浠跺す
        |
        v
spectral-reader  ->  spectral-check  ->  spectral-splitter
                                            |
                                            v
spectral-preprocess  ->  spectral-feature  ->  spectral-modeling
        \                    |                       /
         \                   v                      /
          +---------- spectral-optimizer ----------+
                               |
                               v
                        spectral-report

spectral-workflow锛氳礋璐ｈ矾鐢便€佺‘璁ゃ€佸悎鍚屼紶閫掍笌娴嬭瘯闆嗛殧绂?```

鍏稿瀷鍒嗙被閾捐矾锛?
```text
璇诲彇鏍囧噯鍖?-> 闈炵牬鍧忔€?QC -> 鍒嗗眰鍒掑垎 -> SNV -> PCA/VIP/鍏ㄥ厜璋?-> SVM/PLS-DA/妯″瀷姣旇緝 -> 鏈€缁堟祴璇?-> 璁烘枃鍥?```

鍏稿瀷鍥炲綊閾捐矾锛?
```text
璇诲彇鏍囧噯鍖?-> quality check -> Kennard-Stone/KFold -> MSC/SNV -> PLS-LV/SPA/CARS -> PLSR/SVR/闆嗘垚鍥炲綊 -> 鏈€缁堟祴璇?-> 鍥炲綊鍥?```

## 涔濅釜 Skill 鐨勮缁嗚兘鍔?
### 1. `spectral-reader`锛氬厜璋辨暟鎹鍙栦笌鏍囧噯鍖?
灏嗕笉鍚屾潵婧愬拰甯冨眬鐨勫厜璋辨暟鎹浆鎹负涓嬫父缁熶竴浣跨敤鐨勬爣鍑嗘暟鎹寘銆傚畠鍙礋璐ｂ€滄纭鍏モ€濓紝涓嶅仛 QC銆佹彃琛ャ€佸垹鏍锋湰銆侀澶勭悊鎴栧缓妯°€?
**鏀寔鐨勮緭鍏ユ牸寮?*

- 鏂囨湰琛ㄦ牸锛欳SV銆乀SV銆乀XT锛?- 宸ヤ綔绨匡細Excel銆丱DS锛?- 鏁扮粍/绉戝瀹瑰櫒锛歂PY銆丯PZ銆丮AT锛堥潪 v7.3锛夈€丠DF5銆丯etCDF锛?- 澶栭儴鏍囩銆佺洰鏍囧€笺€佸厓鏁版嵁鍜岀嫭绔嬫尝娈佃酱鏂囦欢锛?- 姣忎釜鏍锋湰涓€涓?CSV/TSV/TXT 鏂囦欢鐨勬枃浠跺す锛?- 鍚厜璋便€佹爣绛俱€佸厓鏁版嵁鍜屾尝娈佃酱鍊欓€夋枃浠剁殑娣峰悎鐩綍銆?
**鏀寔鐨勬暟鎹竷灞€**

- 鏍锋湰鎸夎鎴栫粡纭鍚庢牱鏈寜鍒楋紱
- 鏁板€兼尝闀?娉㈡暟鍒楀悕锛屽 `350鈥?500 nm`銆乣3600鈥?00 cm-1`锛?- 琛ㄥご鍓嶆湁娉ㄩ噴鎴栦华鍣ㄥ鍑鸿鏄庯紱
- 鍏夎氨鍒楀墠鍚庡す鏈夋牱鏈?ID銆佹爣绛俱€佺洰鏍囧€兼垨鍏冩暟鎹紱
- 澶氳琛ㄥご銆丒xcel 澶?sheet銆丯PZ/MAT 鍙橀噺閫夋嫨銆丠DF5/NetCDF dataset path锛?- 鎸?`sample_id` 瀵归綈澶栭儴鏍囩锛屾垨缁忕‘璁ゅ悗鎸夎搴忓榻愶紱
- 浠庢枃浠跺悕鎴栨枃浠跺す鍚嶆彁鍙栨爣绛撅紙蹇呴』鏄惧紡瑕佹眰锛夈€?
**鏍囧噯杈撳嚭**

| 鏂囦欢 | 浣滅敤 |
| --- | --- |
| `X.csv` | 鏁板€煎厜璋辩煩闃碉紝琛屾槸鏍锋湰锛屽垪鏄尝娈?|
| `y.csv` | 鍙€夌殑鍒嗙被鏍囩鎴栧洖褰掔洰鏍?|
| `sample_ids.csv` | 鏍锋湰鍞竴鏍囪瘑 |
| `band_axis.csv` | 娉㈤暱銆佹尝鏁版垨绱㈠紩杞?|
| `metadata.csv` | 鍙€夋牱鏈厓鏁版嵁 |
| `data_contract.json` | 鏁版嵁褰㈢姸銆佹潵婧愩€佸崟浣嶃€佹枃浠跺紩鐢ㄥ拰璀﹀憡 |

閬囧埌澶氫釜鍙В閲婄殑 sheet銆佸彉閲忋€乨ataset銆佹爣绛炬枃浠舵垨鏍锋湰鏂瑰悜鏃讹紝skill 浼氳姹傛渶灏忓繀瑕佺‘璁わ紝涓嶄細闈欓粯鐚滄祴銆?
### 2. `spectral-check`锛氬厜璋辫川閲忔帶鍒?
璇诲彇鏍囧噯鏁版嵁鍖咃紝鍏堟鏌ュ拰鏍囪闂锛涢粯璁や笉鍒犻櫎鏍锋湰鎴栨尝娈点€傚彧鏈夌敤鎴风‘璁ゅ姩浣溿€侀槇鍊煎拰鑼冨洿鍚庯紝鎵嶈繘鍏?`clean` 妯″紡銆?
**榛樿缁煎悎妫€鏌?*

- 鍚堝悓銆佹枃浠躲€佽鍒楀拰娉㈡杞翠竴鑷存€ф鏌ワ紱
- 缂哄け鍊笺€侀潪鏁板€笺€佸父閲忔尝娈靛拰浣庢柟宸尝娈碉紱
- 绫诲埆鏍锋湰閲忋€佺被鍒笉骞宠　鍜屽洖褰掔洰鏍囧紓甯革紱
- 鍏夎氨寮哄害銆佺矖绯欏害銆佸皷宄板拰鍩虹嚎婕傜Щ椋庨櫓锛?- 鍏ㄥ眬/绫诲唴骞冲潎鍏夎氨鐩镐技鎬э紱
- 瀹屽叏閲嶅鍜岃繎閲嶅鍏夎氨锛?- PCA Hotelling T虏銆丵 residual 鍜?PCA 绌洪棿 Mahalanobis 璺濈銆?
**鏀寔鐨勫紓甯告牱鏈柟娉?*

- 绋冲仴 Z 鍒嗘暟锛坄robust_zscore` / robust Z-score锛夛紱
- 鍥涘垎浣嶈窛锛坄iqr` / interquartile range锛夛紱
- 涓綅鏁扮粷瀵瑰亸宸紙`mad` / median absolute deviation锛夛紱
- 鍧囧€肩浉浼兼€э紙`similarity_to_mean`锛夛紱
- 绫诲唴鐩镐技鎬э紙`classwise_similarity`锛夛紱
- 灏栧嘲妫€娴嬶紙`spike_detection`锛夛紱
- 鍩虹嚎婕傜Щ璇勫垎锛坄baseline_drift_score`锛夛紱
- PCA Hotelling T虏锛坄pca_hotelling_t2`锛夛紱
- Q 娈嬪樊锛坄pca_q_residual`锛夛紱
- PCA-Mahalanobis锛坄mahalanobis_on_pca`锛夛紱
- 杩戦噸澶嶆鏌ワ紙`near_duplicate_check`锛夛紱
- 澶氭柟娉曞叡璇嗭紙`multi_method_consensus`锛夛紱
- 鍗婇噰鏍峰紓甯哥ǔ瀹氭€э紙`half_resampling_outlier` / HR锛夛紱
- 钂欑壒鍗℃礇浜ゅ弶楠岃瘉寮傚父绋冲畾鎬э紙`mccv_outlier` / MCCV锛夈€?
HR/MCCV 鍙湪鐢ㄦ埛瑕佹眰楂樼骇寮傚父绋冲畾鎬у垎鏋愭椂杩愯锛涘畠浠瘑鍒殑鏄綋鍓嶇绾夸笅鐨勪笉绋冲畾鏍锋湰锛屼笉绛夊悓浜庤嚜鍔ㄥ垽瀹氣€滃潖鍏夎氨鈥濄€?
**缁忕‘璁ゅ悗鏀寔鐨勬竻鐞嗗姩浣?*

- 鍒犻櫎楂樼疆淇″紓甯告牱鏈紱
- 鍒犻櫎瀹屽叏閲嶅鎴栬繎閲嶅鍏夎氨锛屾垨杈撳嚭鍒嗙粍寤鸿浜ょ粰 splitter锛?- 鍒犻櫎甯搁噺銆侀珮缂哄け鐜囥€佷弗閲嶄綆鏂瑰樊鎴栭珮灏栧嘲棰戠巼娉㈡锛?- 娉㈡鍧囧€?涓綅鏁版彃琛ャ€佺嚎鎬?鏈€杩戦偦鎻掑€硷紱
- Hampel銆佺Щ鍔ㄤ腑浣嶆暟鎴栧眬閮?MAD 灏栧嘲淇锛?- 鏇存柊娓呯悊鍚庣殑鏍囧噯鏁版嵁鍖呭拰 `qc_cleaning_log.json`銆?
涓昏杈撳嚭涓?`qc_result.json`锛涙竻鐞嗗悗杩樹細鐢熸垚 `cleaned_package/`锛屽苟閫氳繃 `next_package_for_downstream` 鎸囧悜姝ｇ‘鐨勪笅娓歌緭鍏ャ€?
### 3. `spectral-splitter`锛氬彲澶嶇幇鏁版嵁鍒掑垎

鍦ㄤ笉澶嶅埗鍏夎氨鐭╅樀鐨勬儏鍐典笅鐢熸垚鏍锋湰褰掑睘鍜屽垝鍒嗗悎鍚岋紝渚涘悗缁墍鏈夐樁娈靛鐢ㄣ€?
| 涓枃鍚嶇О | 鏂规硶浠ｇ爜 / English name | 閫傜敤鍦烘櫙 |
| --- | --- | --- |
| 闅忔満鐣欏嚭 | `random` / random holdout | 鍥炲綊鎴栨棤鍒嗗眰瑕佹眰鐨勬櫘閫氱暀鍑?|
| 鍒嗗眰鐣欏嚭 | `stratified` / stratified holdout | 鍒嗙被浠诲姟锛屽敖閲忎繚鎸佺被鍒瘮渚?|
| 棰勫畾涔夊垝鍒?| `predefined_split` / predefined split | 澶栭儴楠岃瘉闆嗘垨宸叉湁 split 鍒?|
| K 鎶樹氦鍙夐獙璇?| `kfold` / K-fold CV | 甯歌浜ゅ弶楠岃瘉 |
| 鍒嗗眰 K 鎶?| `stratified_kfold` / stratified K-fold CV | 灏忔牱鏈垎绫?|
| 鐣欎竴娉?| `leave_one_out` / LOOCV | 鏋佸皬鏍锋湰鐨勬樉寮忛渶姹?|
| 钂欑壒鍗℃礇閲嶅鐣欏嚭 | `monte_carlo_cv` / Monte Carlo CV | 閲嶅闅忔満璇勪及 |
| 閲嶅闅忔満鍒掑垎 | `repeated_random_split` / repeated random split | 閲嶅 holdout |
| 鍒嗗眰钂欑壒鍗℃礇 | `stratified_monte_carlo_cv` / stratified MCCV | 閲嶅鍒嗙被璇勪及 |
| Kennard鈥揝tone | `kennard_stone` / Kennard鈥揝tone split | 鍩轰簬 X 绌洪棿瑕嗙洊鐨勪唬琛ㄦ€у垝鍒?|
| SPXY | `spxy` / SPXY split | 鍚屾椂鑰冭檻 X 涓庤繛缁?y 鐨勫洖褰掑垝鍒?|
| Duplex | `duplex` / Duplex split | 浠ｈ〃鎬?train/test 鏋勯€?|
| 鍥炲綊鍒嗙鍒嗗眰 | `regression_stratified` / regression-binned split | 杩炵画 y 鍒嗙鍚庡垎灞?|
| y 鍒嗙鍒嗗眰 | `y_binned_stratified` / y-binned split | 杩炵画 y 鐨勫垎浣嶆暟鍒嗙 |
| 鍒嗙粍鍒掑垎 | `group` / group split | 鍚岀粍鏍锋湰涓嶈法闆嗗悎 |
| 鍒嗙粍闃叉硠婕忓垝鍒?| `group_aware` / group-aware split | 鎵规銆佸彈璇曡€呮垨閲嶅娴嬮噺闅旂 |
| 鍒嗗眰鍒嗙粍鍒掑垎 | `stratified_group` / stratified group split | 鍚屾椂淇濇寔绫诲埆鍜岀粍杈圭晫 |

杈撳嚭锛歚split_indices.csv`銆乣split_contract.json` 鍜?`split_summary.json`銆傚綋鍓嶄笉瀹ｇО鏀寔鏃堕棿搴忓垪鍒掑垎銆佹寜鏃堕棿椤哄簭鍒掑垎鎴?nested CV銆?
### 4. `spectral-preprocess`锛氶槻娉勬紡鍏夎氨棰勫鐞?
杈撳叆蹇呴』鍖呭惈 `split_contract.json`銆傞渶瑕佸涔犳€讳綋缁熻閲忕殑鏂规硶鍙湪璁粌闆嗘嫙鍚堬紝鐒跺悗搴旂敤鍒伴獙璇侀泦鍜屾祴璇曢泦锛汣V/閲嶅鐣欏嚭鏃舵寜 fold/repeat 閲嶆柊鎷熷悎銆?
**鏁ｅ皠涓庤秼鍔挎牎姝?*

- 鏃犻澶勭悊锛坄none` / no preprocessing锛夛紱
- 鏍囧噯姝ｆ€佸彉閲忔牎姝ｏ紙`snv` / Standard Normal Variate锛夛紱
- 澶氬厓鏁ｅ皠鏍℃锛坄msc` / Multiplicative Scatter Correction锛夛紱
- 鍘昏秼鍔匡紙`detrend` / detrending锛夛紱
- SNV + 鍘昏秼鍔匡紙`snv_detrend` / SNV plus detrending锛夈€?
**骞虫粦涓庡鏁?*

- Savitzky鈥揋olay 骞虫粦锛坄sg_smoothing`锛夛紱
- 涓€闃跺鏁帮紙`first_derivative`锛夛紱
- 浜岄樁瀵兼暟锛坄second_derivative`锛夛紱
- 绉诲姩骞冲潎锛坄moving_average`锛夛紱
- 楂樻柉骞虫粦锛坄gaussian_smoothing`锛夛紱
- 涓€兼护娉紙`median_filter`锛夈€?
**鍩虹嚎鏍℃**

- 绾挎€у熀绾匡紙`linear_baseline`锛夛紱
- 澶氶」寮忓熀绾匡紙`polynomial_baseline`锛夛紱
- 姗＄毊绛嬪熀绾匡紙`rubberband_baseline`锛夛紱
- 闈炲绉版渶灏忎簩涔樺熀绾匡紙`als_baseline` / asymmetric least squares锛夈€?
**缂╂斁涓庡綊涓€鍖?*

- 鍧囧€间腑蹇冨寲锛坄mean_centering`锛夛紱
- 鏍囧噯鍖栵紙`standardization`锛夛紱
- 鏈€灏忊€撴渶澶х缉鏀撅紙`minmax_scaling`锛夛紱
- 椴佹缂╂斁锛坄robust_scaling`锛夛紱
- Pareto 缂╂斁锛坄pareto_scaling`锛夛紱
- L2 褰掍竴鍖栵紙`l2_normalization`锛夛紱
- 闈㈢Н褰掍竴鍖栵紙`area_normalization`锛夛紱
- 鏈€澶х粷瀵瑰€煎綊涓€鍖栵紙`max_abs_normalization`锛夈€?
**鐗╃悊杞崲涓庢尝娈靛鐞?*

- 鍙嶅皠鐜囪浆鍚稿厜搴︼紙`reflectance_to_absorbance`锛夛紱
- 閫忓皠鐜囪浆鍚稿厜搴︼紙`transmittance_to_absorbance`锛夛紱
- 瀵规暟鍙樻崲锛坄log_transform`锛夛紱
- 淇濈暀鐗╃悊娉㈡鑼冨洿锛坄band_range_select`锛夛紱
- 绉婚櫎鐗╃悊娉㈡鑼冨洿锛坄remove_band_ranges`锛夈€?
MSC銆佷腑蹇冨寲鍜岀缉鏀剧被鏂规硶灞炰簬 train-fit 鏂规硶銆係G/瀵兼暟銆佸熀绾裤€佸惛鍏夊害杞崲鍜屾尝娈佃寖鍥存柟娉曢渶瑕佺‘璁ゅ叧閿弬鏁版垨鐗╃悊鍚箟銆傝緭鍑轰负鏂扮殑鏍囧噯鍖呫€乣preprocess_state.json` 鍜?`preprocess_contract.json`銆?
### 5. `spectral-feature`锛氱壒寰佸伐绋嬨€侀檷缁翠笌鍙橀噺閫夋嫨

姣忔鎵ц涓€涓壒寰佹柟娉曘€傛墍鏈夐渶瑕佹嫙鍚堢殑鏂规硶鍙敤璁粌闆嗭紱娣卞害宓屽叆闇€棰濆纭璁粌棰勭畻鍜屼笓灞炲弬鏁般€?
#### 浼犵粺鐗瑰緛涓庡寲瀛﹁閲忔柟娉?
| 涓枃鍚嶇О | 鏂规硶浠ｇ爜 / English name |
| --- | --- |
| 涓嶅仛鐗瑰緛宸ョ▼ | `none` / no feature engineering |
| 涓绘垚鍒嗗垎鏋?| `pca` / Principal Component Analysis |
| PLS 娼滃彉閲?| `pls_latent_variables` / PLS latent variables |
| VIP 鍙橀噺閲嶈鎬?| `vip` / Variable Importance in Projection |
| KBest 缁熻绛涢€?| `select_k_best` / SelectKBest |
| SPA 杩炵画鎶曞奖绠楁硶 | `spa` / Successive Projections Algorithm |
| CARS 绔炰簤鎬ц嚜閫傚簲閲嶅姞鏉冮噰鏍?| `cars` / Competitive Adaptive Reweighted Sampling |
| UVE 鏃犱俊鎭彉閲忓墧闄?| `uve` / Uninformative Variable Elimination |
| MCUVE 钂欑壒鍗℃礇 UVE | `mcuve` / Monte Carlo UVE |
| 鍖洪棿 PLS | `interval_pls` / interval PLS |
| 鐩稿叧鎬х瓫閫?| `correlation_filter` / correlation filter |
| ANOVA F 绛涢€?| `anova_f` / ANOVA F-test |
| 鍥炲綊 F 绛涢€?| `f_regression` / F-regression |
| 鏂瑰樊闃堝€?| `variance_threshold` / variance threshold |
| 鎸囧畾娉㈡鑼冨洿 | `select_by_band_range` / select by band range |
| 鎸囧畾娉㈡绱㈠紩 | `select_by_band_indices` / select by band indices |

#### 鎶曞奖銆佷俊鍙峰彉鎹笌娴佸舰鏂规硶

| 涓枃鍚嶇О | 鏂规硶浠ｇ爜 / English name | 璇存槑 |
| --- | --- | --- |
| 鏍?PCA | `kernel_pca` / Kernel PCA | 闈炵嚎鎬ф姇褰?|
| 绋€鐤?PCA | `sparse_pca` / Sparse PCA | 绋€鐤忚浇鑽?|
| 闈炶礋鐭╅樀鍒嗚В | `nmf` / NMF | 杈撳叆蹇呴』闈炶礋 |
| 鐙珛鎴愬垎鍒嗘瀽 | `ica_embedding` / ICA | 鐙珛鎴愬垎鎶曞奖 |
| LDA 鐩戠潱鎶曞奖 | `lda_projection` / LDA projection | 鍒嗙被鐩戠潱闄嶇淮 |
| DCT 鐗瑰緛 | `dct_features` / Discrete Cosine Transform | 纭畾鎬ч€愭牱鏈彉鎹?|
| FFT 鐗瑰緛 | `fft_features` / Fast Fourier Transform | 纭畾鎬ч鍩熺壒寰?|
| 瀛楀吀瀛︿範 | `dictionary_learning` / Dictionary Learning | 绋€鐤忓瓧鍏歌〃绀?|
| Isomap 宓屽叆 | `isomap_embedding` / Isomap | 榛樿鎺㈢储鐢ㄩ€旓紱寤烘ā闇€纭 |
| LLE 宓屽叆 | `lle_embedding` / Locally Linear Embedding | 榛樿鎺㈢储鐢ㄩ€旓紱寤烘ā闇€纭 |
| t-SNE 宓屽叆 | `tsne_embedding` / t-SNE | 浠呯敤浜庡彲瑙嗗寲锛屼笉杩涘叆鎬ц兘璇佹槑 |
| UMAP 宓屽叆 | `umap_embedding` / UMAP | 榛樿鎺㈢储鐢ㄩ€旓紱寤烘ā闇€纭 |

#### 娣卞害鍏夎氨宓屽叆

| 涓枃鍚嶇О | 鏂规硶浠ｇ爜 / English name | 鏂规硶涓撳睘纭 |
| --- | --- | --- |
| 鑷紪鐮佸櫒宓屽叆 | `autoencoder_embedding` / Autoencoder embedding | 缁村害銆佽缁冭疆鏁扮瓑 |
| 鍘诲櫔鑷紪鐮佸櫒宓屽叆 | `denoising_autoencoder_embedding` / Denoising AE | `noise_std` |
| 涓€缁?CNN 鍏夎氨宓屽叆 | `cnn_1d_embedding` / 1D CNN spectral embedding | 缃戠粶棰勭畻涓庢鍒欏寲 |
| ResNet1D 鍏夎氨宓屽叆 | `resnet1d_embedding` / ResNet1D embedding | 灏忔牱鏈繃鎷熷悎椋庨櫓 |
| CLS-former 鍏夎氨宓屽叆 | `cls_former_embedding` / CLS-former embedding | `patch_size` |
| 鎺╃爜鍏夎氨鑷紪鐮佸櫒宓屽叆 | `masked_spectral_autoencoder_embedding` / Masked spectral AE | `mask_ratio`, `patch_size` |
| 瀵规瘮鍏夎氨宓屽叆 | `contrastive_spectral_embedding` / Contrastive spectral embedding | `temperature` 鍜屽寮哄己搴?|

娣卞害鏂规硶浼氱‘璁?`n_components`銆乣epochs`銆乣batch_size`銆乣learning_rate`銆乣weight_decay`銆侀殢鏈虹瀛愩€佽澶囧拰鏂规硶涓撳睘鍙傛暟銆傜敤浜庡垎绫荤殑宓屽叆寤鸿姣旇緝 8/16/32 缁达紱2 缁撮€氬父鍙敤浜庡彲瑙嗗寲銆傝瑙夊垎绂讳笉鑳戒唬鏇垮垎绫绘垨鍥炲綊鎸囨爣銆?
杈撳嚭涓烘柊鐨勬爣鍑嗗寘銆乣feature_state.json` 鍜?`feature_contract.json`锛屽苟淇濈暀閫変腑娉㈡銆佽浇鑽枫€佽缁冨璁＄瓑鍙拷婧俊鎭€?
### 6. `spectral-modeling`锛氬垎绫汇€佸洖褰掋€佹ā鍨嬫瘮杈冧笌璋冨弬

杈撳叆鍙互鏄?split 鍚庣殑鏍囧噯鍖呫€乣preprocess_contract.json` 鎴?`feature_contract.json`銆傛ā鍨嬪彧鍦ㄨ缁冮泦鎷熷悎锛屼娇鐢ㄩ獙璇侀泦鎴栬缁冮泦鍐呴儴 CV 閫夋ā鍨?鍙傛暟锛岄攣瀹氬悗鎵嶈闂祴璇曢泦銆?
#### 鍒嗙被妯″瀷

**浼犵粺鏈哄櫒瀛︿範涓庡寲瀛﹁閲忓垎绫诲櫒**

- 閫昏緫鍥炲綊锛坄logistic_regression` / Logistic Regression锛夛紱
- 绾挎€?SVM锛坄linear_svm` / Linear SVM锛夛紱
- RBF 鏀寔鍚戦噺鏈猴紙`svm` / RBF-SVM锛夛紱
- 绾挎€у垽鍒垎鏋愶紙`lda` / Linear Discriminant Analysis锛夛紱
- 浜屾鍒ゅ埆鍒嗘瀽锛坄qda` / Quadratic Discriminant Analysis锛夛紱
- 楂樻柉鏈寸礌璐濆彾鏂紙`gaussian_nb` / Gaussian Naive Bayes锛夛紱
- PLS-DA锛坄pls_da` / Partial Least Squares Discriminant Analysis锛夛紱
- SIMCA锛坄simca` / Soft Independent Modeling of Class Analogy锛夛紱
- K 杩戦偦锛坄knn_classifier` / KNN classifier锛夛紱
- 闅忔満妫灄锛坄random_forest_classifier` / Random Forest classifier锛夛紱
- Extra Trees锛坄extra_trees_classifier` / Extra Trees classifier锛夛紱
- 姊害鎻愬崌锛坄gradient_boosting_classifier` / Gradient Boosting classifier锛夛紱
- 澶氬眰鎰熺煡鏈猴紙`mlp_classifier` / MLP classifier锛夈€?
**鍙€変緷璧?boosting 鍒嗙被鍣?*

- XGBoost锛坄xgboost_classifier`锛夛紱
- LightGBM锛坄lightgbm_classifier`锛夛紱
- CatBoost锛坄catboost_classifier`锛夈€?
**瀹為獙鎬у皬鏍锋湰娣卞害/姒傜巼鍒嗙被鍣?*

- 娣辨牳瀛︿範楂樻柉杩囩▼鍒嗙被鍣紙`spectral_dkl_gp_classifier` / spectral DKL-GP classifier锛夛紱
- 鍘熷瀷鍏夎氨鍒嗙被鍣紙`proto_spectral_classifier` / prototypical spectral classifier锛夛紱
- CLS-former 鍒嗙被鍣紙`cls_former_classifier` / CLS-former classifier锛夛紱
- CLS-former 宓屽叆 + SVM锛坄cls_former_embedding_svm` / CLS-former embedding SVM锛夈€?
甯哥敤蹇€熸瘮杈冮泦 `regular-fast` 鍖呭惈锛氶€昏緫鍥炲綊銆佺嚎鎬?SVM銆丷BF-SVM銆丩DA銆並NN銆侀殢鏈烘．鏋楀拰 Extra Trees銆傚畠鏄€熷害涓庤鐩栭潰鐨勯粯璁ゆ姌涓紝涓嶅寘鍚?QDA銆丟aussian NB銆丳LS-DA銆丼IMCA銆丟radient Boosting銆丮LP銆佸彲閫?boosting 鎴栧疄楠屾繁搴︽ā鍨嬶紱杩欎簺鏂规硶浠嶅彲閫氳繃鑷畾涔夊垪琛ㄦ垨鏇村畬鏁寸殑姣旇緝鏂规绾冲叆銆?
#### 鍥炲綊妯″瀷

**浼犵粺鏈哄櫒瀛︿範涓庡寲瀛﹁閲忓洖褰掑櫒**

- PLS 鍥炲綊锛坄plsr` / Partial Least Squares Regression锛夛紱
- 涓绘垚鍒嗗洖褰掞紙`pcr` / Principal Component Regression锛夛紱
- 绾挎€у洖褰掞紙`linear_regression` / Linear Regression锛夛紱
- 宀洖褰掞紙`ridge` / Ridge Regression锛夛紱
- Lasso锛坄lasso` / Lasso Regression锛夛紱
- 寮规€х綉缁滐紙`elastic_net` / Elastic Net锛夛紱
- 璐濆彾鏂箔鍥炲綊锛坄bayesian_ridge` / Bayesian Ridge锛夛紱
- 鏀寔鍚戦噺鍥炲綊锛坄svr` / Support Vector Regression锛夛紱
- K 杩戦偦鍥炲綊锛坄knn_regressor` / KNN regressor锛夛紱
- 闅忔満妫灄鍥炲綊锛坄random_forest_regressor`锛夛紱
- Extra Trees 鍥炲綊锛坄extra_trees_regressor`锛夛紱
- 姊害鎻愬崌鍥炲綊锛坄gradient_boosting_regressor`锛夛紱
- 楂樻柉杩囩▼鍥炲綊锛坄gpr` / Gaussian Process Regression锛夈€?
**鍙€変緷璧栧洖褰掑櫒**锛歚xgboost_regressor`銆乣lightgbm_regressor`銆乣catboost_regressor`銆?
**瀹為獙鎬у皬鏍锋湰鍥炲綊鍣?*锛歚spectral_dkl_gp_regressor`銆乣proto_spectral_regressor`銆乣cls_former_regressor`銆?
#### 鑷姩璋冨弬鍜岃瘎浼版ā寮?
- **鍥哄畾鍙傛暟**锛氫娇鐢ㄧ敤鎴风粰瀹氭垨鏄庣‘榛樿鍙傛暟锛岄€傚悎蹇€熷熀绾垮拰鍏钩鍥哄畾鍙傛暟姣旇緝锛?- **鍒嗙被鍣?鍥炲綊鍣ㄥ唴閮ㄨ皟鍙?*锛氬彧鏀瑰彉妯″瀷鍙傛暟锛屼娇鐢ㄩ獙璇侀泦鎴栬缁冮泦鍐呴儴 CV锛?- **validation-only**锛氬彧杈撳嚭 train/validation 鎸囨爣锛屼笉璁块棶 test锛岄€傚悎 optimizer trial锛?- **鏈€缁堥攣瀹氭祴璇?*锛氭ā鍨嬪拰鍙傛暟閿佸畾鍚庤瘎浼颁竴娆?test锛?- **confirmatory test**锛氭祴璇曢泦宸茶鏌ョ湅鏃讹紝鏄庣‘鏍囨敞涓虹‘璁ゆ€х粨鏋滆€岄潪瀹屽叏鐩叉祴锛?- **閲嶅鍒掑垎/CV**锛氭寜 fold/repeat 閲嶆嫙鍚堝苟姹囨€伙紝涓嶆妸澶氭缁撴灉璇О涓哄崟娆℃渶缁堟祴璇曘€?
鑷姩璋冨弬涓嶄細鏇跨敤鎴疯嚜鍔ㄦ敼鍙樹笂娓搁澶勭悊鎴栫壒寰佹柟娉曪紱璺ㄩ樁娈佃皟浼樺睘浜?`spectral-optimizer`銆傚疄楠屾ā鍨嬮渶瑕佹樉寮忕‘璁よ缁冮绠楀拰鍏抽敭鍙傛暟銆?
**涓昏杈撳嚭**

- `modeling_contract.json`銆乣modeling_summary.json`銆乣metrics.json`锛?- `predictions.csv`銆乣model_artifact.pkl`锛?- 澶氬垎绫诲櫒楠岃瘉姣旇緝琛?`classifier_validation_summary.csv`锛岄€愯璁板綍 train/validation accuracy銆乥alanced accuracy銆丮acro-F1銆丄UC銆佸弬鏁板拰 `test_accessed`锛?- 鍒嗙被鍙€?`confusion_matrix.csv`锛?- 涓嶇‘瀹氭€фā鍨嬪彲閫?`prediction_std.csv`銆乣uncertainty_summary.json`锛?- 澶氬垎绫诲櫒楠岃瘉姣旇緝杈撳嚭姣忎釜妯″瀷鐨?train/validation 鎸囨爣銆佸弬鏁板拰 `test_accessed` 鐘舵€併€?
### 7. `spectral-optimizer`锛氬彈棰勭畻绾︽潫鐨勮嚜鍔ㄤ紭鍖?
optimizer 璐熻矗鈥滆鍒掑€欓€夈€佽皟鐢ㄥ畼鏂瑰瓙 skill銆佹瘮杈冮獙璇?CV 缁撴灉銆侀€夋嫨鏈€浣崇绾库€濓紝涓嶈嚜琛岄噸鍐欓澶勭悊銆佺壒寰佹垨妯″瀷绠楁硶銆?
**鍥涚妯″紡**

| 妯″紡 | 鍔熻兘 |
| --- | --- |
| `recommend_from_profile` | 鍙牴鎹牱鏈噺銆佺壒寰侀噺銆佷换鍔″拰绫诲埆缁撴瀯鎺ㄨ崘鍊欓€夛紝涓嶈繍琛屾ā鍨?|
| `tune_method` | 鍥哄畾鍏朵粬闃舵锛屼粎璋冧竴涓柟娉曠殑鍙傛暟 |
| `compare_step` | 鍥哄畾涓婁笅娓革紝姣旇緝棰勫鐞嗐€佺壒寰佹垨妯″瀷涓殑涓€涓樁娈?|
| `optimize_pipeline` | 鍦ㄧ‘璁ょ殑鏂规硶绌洪棿鍜?trial 棰勭畻鍐呰仈鍚堟悳绱㈠闃舵绠＄嚎 |

**涓夊眰璋冨弬寮哄害**

- **Level 1锛氫粎妯″瀷鍙傛暟璋冧紭**銆傚浐瀹氶澶勭悊鍜岀壒寰侊紝鍙皟鍒嗙被鍣?鍥炲綊鍣紱
- **Level 2锛氫紶缁熺壒寰?+ 妯″瀷璋冧紭**銆傛瘮杈?PCA銆丳LS-LV銆乂IP銆並Best銆丼PA 绛夊弬鏁颁笌涓嬫父妯″瀷锛?- **Level 3锛氭繁搴﹀祵鍏?+ 妯″瀷璋冧紭**銆傛瘮杈冨祵鍏ョ淮搴︺€佽缁冨弬鏁板拰涓嬫父妯″瀷锛岃绠楅绠楀ぇ锛屽繀椤诲厛棰勮銆佽鍓苟纭銆?
**瀹夊叏瑙勫垯**

- 蹇呴』鏄庣‘鍊欓€夋柟娉曘€佸弬鏁扮綉鏍笺€侀€夋嫨鎸囨爣鍜?`max_trials`锛?- 鎵€鏈?trial 浣跨敤 validation 鎴?train-only CV锛岀姝㈡寜 test 鎸囨爣閫夋柟娉曪紱
- 骞冲垎鏃朵紭鍏堟洿绠€鍗曢澶勭悊銆佹洿灏戣緭鍑虹壒寰併€佹洿灏戣皟鍙傚拰鏇翠綆璁＄畻鎴愭湰锛?- `best_pipeline.json` 淇濆瓨 preprocess + feature + model 鐨勫畬鏁?lineage锛?- 鏈€浣崇绾块攣瀹氬悗鎵嶅彲鍗曠嫭纭鏈€缁堟祴璇曪紱
- 灏?holdout 鐨勯珮鍒嗗缓璁敤 5/10 娆?repeated holdout 澶嶆牳绋冲畾鎬с€?
杈撳嚭锛歚optimizer_contract.json`銆乣optimization_plan.json`銆乣candidate_space.json`銆乣trial_manifest.csv`銆佹墽琛屽悗鍙€?`trial_results.csv`銆乣best_pipeline.json` 鍜?`recommendation_report.md`銆?
### 8. `spectral-report`锛氳鏂囩骇鍥捐〃涓庡彲澶嶇幇鎶ュ憡

浠?reader銆丵C銆乻plitter銆乸reprocess銆乫eature銆乵odeling 鎴?optimizer 鐨勭幇鏈変骇鐗╃敓鎴愬浘琛紱涓嶉噸鏂拌缁冩ā鍨嬶紝涓嶇紪閫犻噸澶嶅疄楠屻€佽宸潯銆佹樉钁楁€ф垨鍗曚綅銆?
**鏀寔鐨勫浘琛ㄧ被鍨?*

- 鍘熷/棰勫鐞嗗厜璋便€佸潎鍊间笌鐪熷疄绂绘暎甯︼紱
- 娉㈡閫夋嫨銆乂IP銆佽浇鑽枫€佺郴鏁板拰娼滃彉閲忚В閲婂浘锛?- PCA銆乁MAP銆乼-SNE銆佹繁搴﹀祵鍏ユ暎鐐瑰浘锛?- 鍒嗙被鍣ㄦ寚鏍囨瘮杈冦€侀噸澶嶅疄楠岀绾垮浘/鐐硅寖鍥村浘锛?- 娣锋穯鐭╅樀銆丷OC銆丳R銆佹牎鍑嗘洸绾匡紱
- 鍥炲綊棰勬祴鍊尖€撳疄娴嬪€笺€?:1 绾裤€佹畫宸浘鍜屾寚鏍囬潰鏉匡紱
- optimizer 鍊欓€夋帓鍚嶃€乼rial landscape 鍜岄攣瀹氱绾挎€ц兘锛?- 鐜版湁鍥剧墖鐨勫璁′笌閲嶇粯銆?
**榛樿璁烘枃椋庢牸**

- 鐧藉簳銆佹棤缃戞牸銆佸畬鏁撮粦鑹插叏杈规锛?- Times New Roman锛?- 浣庨ケ鍜屻€侀珮鍖哄垎搴﹁鏂囬厤鑹诧紝涓嶉粯璁ら珮楗卞拰绾壊鎴?hatch锛?- 澶氶潰鏉跨粺涓€浣跨敤澶栫疆灏忓啓鏍囩 `(a)`, `(b)`, 鈥︼紱
- 鏌辩姸鍥句粠闆跺紑濮嬶紝鍗曟 holdout 涓嶄吉閫犺宸潯锛?- 閲嶅缁撴灉灞曠ず鐪熷疄 folds/repeats 鐨勫垎甯冦€乵ean 卤 SD 鎴栧悎閫傜殑涓嶇‘瀹氭€э紱
- caption 璇存槑鏁版嵁鍒嗗尯銆佺粺璁″崟浣嶃€侀噸澶嶅畾涔夊拰鍥惧舰缂栫爜锛?- embedding 鍧愭爣涓嶈兘璺ㄦ柟娉曠洿鎺ユ瘮杈冿紝瑙嗚鍒嗙涓嶈兘浣滀负鍒嗙被鎬ц兘璇佹嵁銆?
**鍥句欢鍖呰緭鍑?*

```text
report_contract.json
figures/*.svg
figures/*.pdf
figures/*.png
source_data/*.csv
code/*.py
captions/*.md
qa/*.md
```

姣忓紶鍥惧繀椤荤粡杩囧昂瀵搞€佽鍒囥€佸瓧浣撱€佸浘渚嬨€佸叏杈规銆佺綉鏍笺€佹暟鎹拷婧拰鏈€缁堟覆鏌?QA銆?
### 9. `spectral-workflow`锛氬闃舵璺敱涓庡悎鍚岀紪鎺?
鏍规嵁鐢ㄦ埛鐩爣閫夋嫨鏈€灏忕殑 child skill 閾撅紝淇濆瓨鍐崇瓥骞朵紶閫掑悎鍚屻€傚畠璐熻矗娴佺▼锛屼笉閲嶅瀹炵幇鍚勯樁娈电畻娉曘€?
**鏀寔鐨勫吀鍨嬭矾绾?*

- 鍙鍙栧拰 QC锛?- 鎺ㄨ崘鍒嗙被/鍥炲綊鍩虹嚎锛?- 鎵嬪姩閫愰樁娈甸€夋嫨锛?- 棰勫鐞嗐€佺壒寰佹垨鍒嗙被鍣ㄥ崟闃舵姣旇緝锛?- 浼犵粺澶氶樁娈典紭鍖栵紱
- 娣卞害宓屽叆 + 浼犵粺妯″瀷锛?- 瀹為獙鎬х鍒扮灏忔牱鏈ā鍨嬶紱
- 宸叉湁涓棿 contract 缁窇锛?- 浠庣粨鏋滃悎鍚岃矾鐢卞埌璁烘枃缁樺浘銆?
褰撶敤鎴峰彧璇粹€滃鐞嗚繖涓厜璋辨暟鎹€濇椂锛寃orkflow 鍏堝彧璇绘鏌ユ暟鎹粨鏋勶紝鍐嶇粰鍑鸿矾绾块€夋嫨锛屼笉浼氱珛鍗虫浛鐢ㄦ埛杩愯浠绘剰妯″瀷銆傝繘鍏ユ墜鍔ㄦ祦绋嬪悗锛屾瘡涓樁娈靛簲鏄剧ず鈥滄帹鑽愰」 + 瀹屾暣鏀寔鑿滃崟 + 鍙墽琛岄€夋嫨绀轰緥鈥濄€?
**鍚堝悓閾?*

| 闃舵 | 涓昏杈撳叆 | 涓昏鍚堝悓/杈撳嚭 |
| --- | --- | --- |
| Reader | 鍘熷鏂囦欢/鏂囦欢澶?| `data_contract.json` |
| Check | 鏍囧噯鏁版嵁鍖?| `qc_result.json`锛屽彲閫夋竻鐞嗗寘 |
| Splitter | 鏍囧噯鏁版嵁鍖?| `split_contract.json` |
| Preprocess | 鏁版嵁鍚堝悓 + 鍒掑垎鍚堝悓 | `preprocess_contract.json` |
| Feature | 鏁版嵁/棰勫鐞嗗悎鍚?+ 鍒掑垎鍚堝悓 | `feature_contract.json` |
| Modeling | 鏁版嵁/棰勫鐞?鐗瑰緛鍚堝悓 + 鍒掑垎鍚堝悓 | `modeling_contract.json` |
| Optimizer | 宸茬‘璁ゅ€欓€夌┖闂?+ 涓婃父鍚堝悓 | `optimizer_contract.json`, `best_pipeline.json` |
| Report | 浠讳竴宸插瓨鍦ㄧ粨鏋滃悎鍚?| `report_contract.json` + 鍥句欢鍖?|
| Workflow | 鐢ㄦ埛鐩爣 + 鍚勯樁娈靛悎鍚?| `workflow_plan.json`, `workflow_result.json` |

## 瀹夎

### Codex 鎻掍欢甯傚満

Before importing or enabling the plugin, validate the user-level Codex config:

```bash
python install/check_codex_config.py --json
```

If Codex reports `Could not load config.toml` or `unclosed table, expected ]`,
the failing file is `~/.codex/config.toml`, not the Spectral Skills runtime.
Plugin import makes Codex reload global config, so an old malformed
`[projects.'...']` entry can surface at that moment. Fix or remove the malformed
table, rerun the preflight, and then retry plugin import.

```bash
codex plugin marketplace add https://github.com/XY041216/spectral-skills.git --ref main
codex plugin add spectral-skills@spectral-skills-local-marketplace
```

If the Codex CLI cannot run on Windows, or if the marketplace/plugin entries
exist in `config.toml` but no `spectral-skills` folder appears under
`%USERPROFILE%\.codex\plugins\cache`, run the local installer from the clone
root:

```bash
python install/install_codex_plugin.py --json
```

It validates `config.toml`, writes a backup before changing it, enables the
local marketplace, and materializes the release image into Codex's plugin cache:

```text
%USERPROFILE%\.codex\plugins\cache\spectral-skills-local-marketplace\spectral-skills\<version>\
```

Restart Codex in a new thread after the installer reports success.

Codex Desktop 涔熷彲娣诲姞鑷畾涔夋彃浠跺競鍦猴細

- Marketplace source锛歚https://github.com/XY041216/spectral-skills.git`
- Branch/ref锛歚main`
- Plugin锛歚spectral-skills`

浠撳簱鏍圭洰褰曞悓鏃舵彁渚?`.codex-plugin/plugin.json` 鍜?`.mcp.json`锛屼綔涓?GitHub
鎻掍欢瀵煎叆鍏ュ彛锛涜鍏ュ彛浼氭妸 Codex skills 鎸囧悜宸叉瀯寤虹殑
`plugins/spectral-skills/skills` 鍙戝竷闀滃儚锛岃€屼笉鏄牴鐩綍涓嬬殑寮€鍙戞簮 `skills/`銆?
涓嶈浣跨敤 Codex 鐨?**GitHub skill installer** 鐩存帴浠庝粨搴撳鍏ュ瓙 skill銆係pectral
Skills 鏄竴涓彃浠跺彂甯冨崟鍏冿紝涓嶆槸涓€缁勫彲浠ョ嫭绔嬪畨瑁呯殑 skill 鏂囦欢澶癸紱濡傛灉瀵煎叆鏃ュ織閲屽嚭鐜?`skill-installer` 鎴?`install-skill-from-github.py --repo XY041216/spectral-skills`锛?搴斿仠姝㈣娴佺▼骞舵敼鐢ㄤ笂闈㈢殑鎻掍欢甯傚満瀹夎鍛戒护銆傜洿鎺?skill 瀹夎浼氫涪澶辨彃浠跺厓鏁版嵁銆丮CP
閰嶇疆銆佸叡浜繍琛屾椂鍜屽彂甯冩鏌ワ紝涓嶆槸鍙楁敮鎸佺殑鍙戣褰㈡€併€?
瀹夎鍚庡紑鍚柊浼氳瘽锛屼緥濡傦細

```text
浣跨敤 $spectral-skills:spectral-workflow 澶勭悊 Tablet_ext_0-3.csv锛?璇诲彇銆丵C銆佸垎灞?6:2:2 鍒掑垎銆丼NV銆乫eature=none锛屽苟姣旇緝 SVM 涓?PLS-DA銆?```

### Claude-compatible agents

浠撳簱鍦?`.claude-plugin/` 涓彁渚?Claude-compatible plugin metadata锛?
```bash
claude plugin marketplace add XY041216/spectral-skills
claude plugin install spectral-skills@spectral-skills
```

涓嶆敮鎸佹彃浠跺競鍦虹殑鏅鸿兘浣撳彲浠?clone 浠撳簱锛屽苟灏?`plugins/spectral-skills/` 涓庣浉搴?`SKILL.md` 浣滀负鎶€鑳藉叆鍙ｃ€俙shared/`銆乣spectral_core/` 鍜?`scripts/` 蹇呴』涓?skills 涓€鍚屼繚鐣欍€?
### 鏈湴鑴氭湰杩愯

```bash
git clone https://github.com/XY041216/spectral-skills.git
cd spectral-skills
pip install -r requirements.txt
```

绀轰緥锛?
```bash
python plugins/spectral-skills/skills/spectral-workflow/scripts/run_spectral_workflow.py \
  --input path/to/data.csv \
  --output-dir outputs/workflow_demo \
  --task-goal classification \
  --split-ratio 6:2:2 \
  --preprocess-methods snv \
  --feature-method none \
  --models svm \
  --json
```

鏈湴 marketplace 閰嶇疆瑙?[`install.md`](install.md)銆?
## 鍙€変緷璧?
鍩虹渚濊禆瑙?`requirements.txt`銆備互涓嬭矾寰勯渶瑕佸搴旀湰鍦颁緷璧栵細

- UMAP锛歚umap-learn`锛?- 娣卞害宓屽叆鍜屽疄楠屾繁搴︽ā鍨嬶細PyTorch锛?- XGBoost锛歚xgboost`锛?- LightGBM锛歚lightgbm`锛?- CatBoost锛歚catboost`銆?
缂哄皯鍙€変緷璧栨椂锛岀浉鍏虫柟娉曚細闃绘鎵ц骞惰鏄庣己澶遍」锛屼笉浼氶潤榛樻浛鎹㈡垚鍏朵粬绠楁硶銆?
## 浣跨敤绀轰緥

```text
鐢?spectral-reader 鎶婅繖涓?Excel 鍏夎氨鏂囦欢杞垚鏍囧噯鏁版嵁鍖咃紝鏍锋湰鍦ㄨ锛屾爣绛惧湪 Sheet2銆?```

```text
瀵规爣鍑嗗厜璋卞寘鍋氶潪鐮村潖鎬?QC锛屾爣璁板皷宄般€佸熀绾挎紓绉汇€侀噸澶嶅厜璋卞拰 PCA T虏/Q 寮傚父鍊欓€夛紝涓嶅垹闄ゆ牱鏈€?```

```text
浣跨敤 stratified_monte_carlo_cv 鍋?10 娆?7:3 閲嶅鍒嗙被鍣ㄦ瘮杈冿紝鍥哄畾 SNV + PCA(10)锛屾瘮杈?regular-fast 妯″瀷骞剁粯鍥俱€?```

```text
姣旇緝 PCA(10)銆丳LS-LV(3)銆乂IP(100)銆並Best(80)銆丼PA(80)锛屼娇鐢ㄩ獙璇侀泦 Macro-F1 閫夋嫨锛屼笉璁块棶娴嬭瘯闆嗐€?```

```text
璁粌 8/16/32 缁?contrastive spectral embedding锛屽苟鍦ㄨ缁?楠岃瘉鍐呴儴璋?SVM锛涢攣瀹氱粍鍚堝悗鍐嶇‘璁ゆ渶缁堟祴璇曘€?```

```text
鏍规嵁 modeling_contract 鐢熸垚鍒嗙被姣旇緝鍥俱€佹贩娣嗙煩闃点€乻ource data銆乧aption 鍜?figure QA銆?```

## 浠撳簱缁撴瀯

```text
.agents/                         Codex marketplace metadata
.codex-plugin/                   GitHub plugin-import entrypoint for Codex
.mcp.json                        Root MCP entrypoint delegated to the plugin image
.claude-plugin/                  Claude-compatible metadata
install/                         plugin 鏋勫缓涓庡彂甯冩鏌?plugins/spectral-skills/         鍙戝竷鐢ㄦ彃浠堕暅鍍忥紙鐢辨瀯寤鸿剼鏈敓鎴愶級
skills/                          寮€鍙戞簮锛涗笉鏄?Codex 鍙戝竷瀹夎鍏ュ彛
spectral_core/                   鍏变韩杩愯鏃跺疄鐜?scripts/                         缁熶竴 CLI/runtime 鑴氭湰
README.md                        闈㈠悜鐢ㄦ埛鐨勫畬鏁磋兘鍔涜鏄?install.md                       瀹夎缁嗚妭
```

寮€鍙戞簮浣嶄簬 `skills/` 鍜?`spectral_core/`锛沗plugins/spectral-skills/` 鏄敓鎴愮殑
Codex/Claude 鍙戝竷闀滃儚锛屼笉搴旀墜宸ョ紪杈戙€傚彂甯冨墠杩愯 `install/build_codex_plugin.py`
閲嶆柊鐢熸垚鏍规彃浠跺叆鍙ｅ拰鍙戝竷闀滃儚锛屽苟鐢?`install/check_codex_plugin.py` 楠岃瘉銆?
## 鑳藉姏杈圭晫涓庣瀛﹁В閲?
- 涓嶉潤榛樺垹闄ゆ牱鏈€佹尝娈垫垨鏍囩锛?- 涓嶄娇鐢ㄦ祴璇曢泦閫夋嫨棰勫鐞嗐€佺壒寰併€佹ā鍨嬫垨瓒呭弬鏁帮紱
- 涓嶅惎鍔ㄦ湭纭銆佹棤棰勭畻涓婇檺鐨?AutoML锛?- 涓嶆妸浜岀淮 embedding 鐨勮瑙夊垎绂诲綋浣滃垎绫绘€ц兘锛?- 涓嶆妸鍗曟 holdout 缁撴灉鎻忚堪鎴?repeated holdout 鎴栫ǔ瀹氭€х粨璁猴紱
- 涓嶇紪閫犺宸潯銆佹樉钁楁€с€佺疆淇″尯闂淬€侀噸澶嶆鏁版垨娉㈡鍗曚綅锛?- 涓嶇洿鎺ヨ鍙栨湭瀵煎嚭鐨勪笓鏈変华鍣ㄤ簩杩涘埗鏍煎紡锛涘簲鍏堝鍑轰负鍙楁敮鎸佺殑琛ㄦ牸鎴栧鍣ㄦ牸寮忋€?
## 寮€鍙戜笌鍙戣妫€鏌?
鍦ㄤ粨搴撴牴鐩綍杩愯锛?
```powershell
python -m pytest -q
python install/build_codex_plugin.py --clean --verify --json
python install/check_codex_plugin.py --json
```

鍙戝竷鍓嶄笉瑕佹彁浜ょ紦瀛樼洰褰曘€佷复鏃惰緭鍑恒€佹湰鍦拌櫄鎷熺幆澧冦€佽皟璇曟。妗堟垨涓存椂 QA 杩愯銆備慨鏀规簮 skill 鍚庯紝搴旈噸鏂版瀯寤烘彃浠堕暅鍍忥紝纭繚寮€鍙戞簮涓庡叕寮€鎻掍欢涓€鑷淬€?
## License

鏈」鐩娇鐢?[MIT License](LICENSE)銆?