<!-- research-repo-tools/run-report/v3 -->
# MCMC stepping benchmark timings by trial

Run: `8b1c72b02458bd215d6eff0f58dfb8720bcd807ac7238e440e9146e3e6c104fa`

Evidence: [run payload](runs/runs/8b1c72b02458bd215d6eff0f58dfb8720bcd807ac7238e440e9146e3e6c104fa/run.json) ·
[evidence envelope](runs/runs/8b1c72b02458bd215d6eff0f58dfb8720bcd807ac7238e440e9146e3e6c104fa/evidence.json).

Acquisition schedule: alternating; trials: 2; seed: None.
Each trial is retained separately. Within-phase samples are not independent trials; no pooled significance claim is made.

## Trial 1: baseline → current

Collection: 1 segment(s); 0 abandoned attempt(s). Validated abandoned phases are retained and excluded from accepted pairs; failed-output capture status is
recorded.
These are completed campaign slots; recovery does not establish uninterrupted collection, independent trials, or absence of optional-stopping bias.

- Collection segment `ed3e91f2d9d645568df20a43452d0227` started: 1795551030572183000 (Unix ns).
- stepping/baseline: 1795551057951999000–1795551264100612000 (Unix ns).
- stepping/current: 1795551266218666000–1795551474263537000 (Unix ns).

Ratios are point estimates within one accepted paired suite. Marginal timing intervals do not establish significance.

## stepping

Semantic contract: mcmc-stepping-v1. Collection session: `afb5fc47ba4e40ccbe096a6335cf6cda`.
Collection interval (Unix ns): 1795551030572718000–1795551481776546000.
Acquisition: newly collected.

- baseline: available — inside declared release interval; release v0.5.2; revision `389039f123e989afd20484969eca650017c08d92`.
  Source SHA-256: `ed5cd272f34cb0972584a7dd3f368545a09c71abfdeb621739e259cb872a6407`; harness SHA-256:
  `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`; scientific gate: passed.
- current: available — inside declared release interval; release v0.5.3; revision `3fcdcbb739b839531bf28a2cf247aaa69f3dff2e`.
  Source SHA-256: `cea2e6f5be27fd7fab41fb373161ca3f441b7dc90137106e16fadb25e01c9f9a`; harness SHA-256:
  `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`; scientific gate: passed.

### Mean (ns)

<table>
<thead>
<tr>
<th>Workload</th>
<th>Baseline</th>
<th>Current</th>
<th>Baseline / current</th>
</tr>
</thead>
<tbody>
<tr>
<td>chain/step&#95;by&#95;value</td>
<td>16.9238 ns [16.9106, 16.9383] (0.95 confidence)</td>
<td>17.7361 ns [17.6557, 17.8401] (0.95 confidence)</td>
<td>0.954203</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;accept&#95;reflection</td>
<td>10.423 ns [10.4035, 10.4415] (0.95 confidence)</td>
<td>12.4833 ns [11.8088, 13.2016] (0.95 confidence)</td>
<td>0.834954</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;no&#95;plan</td>
<td>0.726485 ns [0.725736, 0.727345] (0.95 confidence)</td>
<td>0.724913 ns [0.724083, 0.725842] (0.95 confidence)</td>
<td>1.00217</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;reject&#95;reflection</td>
<td>10.642 ns [10.6162, 10.6682] (0.95 confidence)</td>
<td>11.0162 ns [10.9957, 11.041] (0.95 confidence)</td>
<td>0.966031</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;accept</td>
<td>13.9745 ns [13.956, 13.9937] (0.95 confidence)</td>
<td>14.3556 ns [14.2658, 14.5181] (0.95 confidence)</td>
<td>0.973451</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;reject&#95;rollback</td>
<td>200.324 ns [200.015, 200.636] (0.95 confidence)</td>
<td>198.113 ns [197.717, 198.521] (0.95 confidence)</td>
<td>1.01116</td>
</tr>
<tr>
<td>observing/manual&#95;online&#95;sum&#95;100</td>
<td>1279.4 ns [1277.62, 1281.17] (0.95 confidence)</td>
<td>1362.46 ns [1359.08, 1365.95] (0.95 confidence)</td>
<td>0.939038</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;buffer&#95;100</td>
<td>1722.62 ns [1718.92, 1727.13] (0.95 confidence)</td>
<td>1754.97 ns [1752.35, 1757.4] (0.95 confidence)</td>
<td>0.981567</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;binning&#95;100</td>
<td>2315.76 ns [2311.67, 2320.44] (0.95 confidence)</td>
<td>2874.31 ns [2867.87, 2881.16] (0.95 confidence)</td>
<td>0.805675</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;online&#95;stats&#95;100</td>
<td>1466.16 ns [1463.9, 1468.57] (0.95 confidence)</td>
<td>1762.17 ns [1758.98, 1765.35] (0.95 confidence)</td>
<td>0.83202</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;100</td>
<td>1688.1 ns [1684.47, 1691.95] (0.95 confidence)</td>
<td>1729.17 ns [1726.14, 1732.26] (0.95 confidence)</td>
<td>0.976253</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/1</td>
<td>1768.63 ns [1766.19, 1771.16] (0.95 confidence)</td>
<td>1818.8 ns [1814.03, 1823.87] (0.95 confidence)</td>
<td>0.972413</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/16</td>
<td>1778.47 ns [1774.96, 1782.14] (0.95 confidence)</td>
<td>1832.01 ns [1827.12, 1836.75] (0.95 confidence)</td>
<td>0.970773</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/2</td>
<td>1761.65 ns [1760.31, 1763] (0.95 confidence)</td>
<td>1829.98 ns [1825.51, 1834.49] (0.95 confidence)</td>
<td>0.96266</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;reflection&#95;100</td>
<td>864.019 ns [863.154, 864.887] (0.95 confidence)</td>
<td>927.026 ns [925.718, 928.369] (0.95 confidence)</td>
<td>0.932033</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/1</td>
<td>1092.4 ns [1091.25, 1093.61] (0.95 confidence)</td>
<td>1136.8 ns [1132.72, 1140.89] (0.95 confidence)</td>
<td>0.960942</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/16</td>
<td>1066.62 ns [1064.72, 1068.81] (0.95 confidence)</td>
<td>1127.33 ns [1124.09, 1130.59] (0.95 confidence)</td>
<td>0.946146</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/2</td>
<td>1083.71 ns [1082.24, 1085.24] (0.95 confidence)</td>
<td>1133.34 ns [1129.51, 1137.07] (0.95 confidence)</td>
<td>0.956213</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;100</td>
<td>1294.65 ns [1293.28, 1296.02] (0.95 confidence)</td>
<td>1348.24 ns [1346.43, 1350.08] (0.95 confidence)</td>
<td>0.960258</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/1</td>
<td>1116.66 ns [1114.27, 1119.34] (0.95 confidence)</td>
<td>1133.7 ns [1128.42, 1138.98] (0.95 confidence)</td>
<td>0.984967</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/16</td>
<td>1087.49 ns [1086.02, 1089.14] (0.95 confidence)</td>
<td>1101.94 ns [1098.51, 1105.38] (0.95 confidence)</td>
<td>0.986884</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/2</td>
<td>1086.12 ns [1085.07, 1087.13] (0.95 confidence)</td>
<td>1109.09 ns [1105.09, 1113.03] (0.95 confidence)</td>
<td>0.979288</td>
</tr>
</tbody>
</table>

### Median (ns)

<table>
<thead>
<tr>
<th>Workload</th>
<th>Baseline</th>
<th>Current</th>
<th>Baseline / current</th>
</tr>
</thead>
<tbody>
<tr>
<td>chain/step&#95;by&#95;value</td>
<td>16.9114 ns [16.9032, 16.9261] (0.95 confidence)</td>
<td>17.6295 ns [17.6111, 17.6529] (0.95 confidence)</td>
<td>0.959268</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;accept&#95;reflection</td>
<td>10.4209 ns [10.4091, 10.4377] (0.95 confidence)</td>
<td>10.9795 ns [10.9573, 10.9919] (0.95 confidence)</td>
<td>0.949128</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;no&#95;plan</td>
<td>0.725762 ns [0.725236, 0.726314] (0.95 confidence)</td>
<td>0.72456 ns [0.723814, 0.725005] (0.95 confidence)</td>
<td>1.00166</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;reject&#95;reflection</td>
<td>10.6111 ns [10.5948, 10.6318] (0.95 confidence)</td>
<td>11.0204 ns [11.0024, 11.0313] (0.95 confidence)</td>
<td>0.96286</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;accept</td>
<td>13.9542 ns [13.9152, 14.0083] (0.95 confidence)</td>
<td>14.2919 ns [14.2717, 14.3093] (0.95 confidence)</td>
<td>0.976374</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;reject&#95;rollback</td>
<td>200.219 ns [199.861, 200.84] (0.95 confidence)</td>
<td>198.274 ns [197.765, 198.668] (0.95 confidence)</td>
<td>1.00981</td>
</tr>
<tr>
<td>observing/manual&#95;online&#95;sum&#95;100</td>
<td>1278.22 ns [1277.18, 1279.52] (0.95 confidence)</td>
<td>1362.3 ns [1358.06, 1366.43] (0.95 confidence)</td>
<td>0.938286</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;buffer&#95;100</td>
<td>1717.69 ns [1715.63, 1720.7] (0.95 confidence)</td>
<td>1756.6 ns [1754.87, 1757.87] (0.95 confidence)</td>
<td>0.977853</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;binning&#95;100</td>
<td>2311.01 ns [2307.8, 2314.7] (0.95 confidence)</td>
<td>2874.65 ns [2866.72, 2881.43] (0.95 confidence)</td>
<td>0.803929</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;online&#95;stats&#95;100</td>
<td>1464.68 ns [1462.27, 1467.46] (0.95 confidence)</td>
<td>1761.65 ns [1758.73, 1765.77] (0.95 confidence)</td>
<td>0.831425</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;100</td>
<td>1683.97 ns [1680.65, 1687.83] (0.95 confidence)</td>
<td>1727.66 ns [1725.61, 1731.31] (0.95 confidence)</td>
<td>0.974714</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/1</td>
<td>1768.14 ns [1764.7, 1770.8] (0.95 confidence)</td>
<td>1816.82 ns [1811.6, 1821.32] (0.95 confidence)</td>
<td>0.973204</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/16</td>
<td>1778.89 ns [1772.14, 1780.84] (0.95 confidence)</td>
<td>1833.28 ns [1827.72, 1838.95] (0.95 confidence)</td>
<td>0.97033</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/2</td>
<td>1761.66 ns [1760.79, 1763.84] (0.95 confidence)</td>
<td>1829.7 ns [1825.77, 1835.16] (0.95 confidence)</td>
<td>0.962811</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;reflection&#95;100</td>
<td>864.058 ns [863.535, 864.697] (0.95 confidence)</td>
<td>927.872 ns [925.499, 928.739] (0.95 confidence)</td>
<td>0.931225</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/1</td>
<td>1091.85 ns [1090.97, 1092.77] (0.95 confidence)</td>
<td>1143.12 ns [1127.44, 1148.96] (0.95 confidence)</td>
<td>0.955144</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/16</td>
<td>1063.25 ns [1062.34, 1064.91] (0.95 confidence)</td>
<td>1125.39 ns [1118.13, 1133.85] (0.95 confidence)</td>
<td>0.944786</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/2</td>
<td>1082.47 ns [1080.96, 1084.89] (0.95 confidence)</td>
<td>1138.29 ns [1122.36, 1142.74] (0.95 confidence)</td>
<td>0.950961</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;100</td>
<td>1294.53 ns [1293.71, 1294.91] (0.95 confidence)</td>
<td>1347.89 ns [1346.64, 1349.66] (0.95 confidence)</td>
<td>0.960413</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/1</td>
<td>1116.39 ns [1113.67, 1118.38] (0.95 confidence)</td>
<td>1144.46 ns [1117.16, 1151.38] (0.95 confidence)</td>
<td>0.975467</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/16</td>
<td>1086.36 ns [1085.56, 1087.34] (0.95 confidence)</td>
<td>1104.01 ns [1097.56, 1107.64] (0.95 confidence)</td>
<td>0.984017</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/2</td>
<td>1086.37 ns [1085.43, 1086.77] (0.95 confidence)</td>
<td>1115.52 ns [1096.36, 1123.06] (0.95 confidence)</td>
<td>0.973874</td>
</tr>
</tbody>
</table>

## Trial 2: current → baseline

Collection: 1 segment(s); 0 abandoned attempt(s). Validated abandoned phases are retained and excluded from accepted pairs; failed-output capture status is
recorded.
These are completed campaign slots; recovery does not establish uninterrupted collection, independent trials, or absence of optional-stopping bias.

- Collection segment `9040d12f2d314a68b43a1f5a1ab9f697` started: 1795551514024007000 (Unix ns).
- stepping/current: 1795551541369307000–1795551752455639000 (Unix ns).
- stepping/baseline: 1795551754440805000–1795551959549079000 (Unix ns).

Ratios are point estimates within one accepted paired suite. Marginal timing intervals do not establish significance.

## stepping

Semantic contract: mcmc-stepping-v1. Collection session: `c0e473bb9df6413db4b73459e6931508`.
Collection interval (Unix ns): 1795551514024573000–1795551963374008000.
Acquisition: newly collected.

- baseline: available — inside declared release interval; release v0.5.2; revision `389039f123e989afd20484969eca650017c08d92`.
  Source SHA-256: `ed5cd272f34cb0972584a7dd3f368545a09c71abfdeb621739e259cb872a6407`; harness SHA-256:
  `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`; scientific gate: passed.
- current: available — inside declared release interval; release v0.5.3; revision `3fcdcbb739b839531bf28a2cf247aaa69f3dff2e`.
  Source SHA-256: `cea2e6f5be27fd7fab41fb373161ca3f441b7dc90137106e16fadb25e01c9f9a`; harness SHA-256:
  `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`; scientific gate: passed.

### Mean (ns)

<table>
<thead>
<tr>
<th>Workload</th>
<th>Baseline</th>
<th>Current</th>
<th>Baseline / current</th>
</tr>
</thead>
<tbody>
<tr>
<td>chain/step&#95;by&#95;value</td>
<td>16.6044 ns [16.5734, 16.6332] (0.95 confidence)</td>
<td>17.2472 ns [17.2371, 17.2581] (0.95 confidence)</td>
<td>0.962732</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;accept&#95;reflection</td>
<td>10.4591 ns [10.435, 10.4836] (0.95 confidence)</td>
<td>12.4568 ns [11.8002, 13.1511] (0.95 confidence)</td>
<td>0.839633</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;no&#95;plan</td>
<td>0.721063 ns [0.720009, 0.722291] (0.95 confidence)</td>
<td>0.72939 ns [0.728727, 0.730172] (0.95 confidence)</td>
<td>0.988585</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;reject&#95;reflection</td>
<td>10.639 ns [10.6122, 10.6645] (0.95 confidence)</td>
<td>11.0811 ns [11.0627, 11.1025] (0.95 confidence)</td>
<td>0.960104</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;accept</td>
<td>13.8895 ns [13.8561, 13.9221] (0.95 confidence)</td>
<td>14.3338 ns [14.2331, 14.5118] (0.95 confidence)</td>
<td>0.969004</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;reject&#95;rollback</td>
<td>192.497 ns [192.052, 192.938] (0.95 confidence)</td>
<td>199.352 ns [199.056, 199.639] (0.95 confidence)</td>
<td>0.965615</td>
</tr>
<tr>
<td>observing/manual&#95;online&#95;sum&#95;100</td>
<td>1262.03 ns [1259.92, 1264.16] (0.95 confidence)</td>
<td>1356.41 ns [1353.55, 1359.31] (0.95 confidence)</td>
<td>0.930422</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;buffer&#95;100</td>
<td>1693.48 ns [1690.22, 1696.76] (0.95 confidence)</td>
<td>1751.88 ns [1748.57, 1754.63] (0.95 confidence)</td>
<td>0.966669</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;binning&#95;100</td>
<td>2283.35 ns [2280.79, 2286] (0.95 confidence)</td>
<td>2826.2 ns [2822.18, 2830.08] (0.95 confidence)</td>
<td>0.807925</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;online&#95;stats&#95;100</td>
<td>1441.29 ns [1438.28, 1444.2] (0.95 confidence)</td>
<td>1748.01 ns [1744.51, 1751.5] (0.95 confidence)</td>
<td>0.824531</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;100</td>
<td>1653.03 ns [1650.95, 1654.98] (0.95 confidence)</td>
<td>1771.71 ns [1769.01, 1774.64] (0.95 confidence)</td>
<td>0.933016</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/1</td>
<td>1734.34 ns [1731.28, 1737.59] (0.95 confidence)</td>
<td>1812.07 ns [1807.26, 1816.84] (0.95 confidence)</td>
<td>0.957108</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/16</td>
<td>1730.99 ns [1726.41, 1736.29] (0.95 confidence)</td>
<td>1821.76 ns [1815.91, 1827.06] (0.95 confidence)</td>
<td>0.95017</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/2</td>
<td>1729.85 ns [1727.08, 1732.44] (0.95 confidence)</td>
<td>1830 ns [1824.64, 1835.17] (0.95 confidence)</td>
<td>0.945274</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;reflection&#95;100</td>
<td>851.073 ns [849.833, 852.366] (0.95 confidence)</td>
<td>934.309 ns [932.981, 935.614] (0.95 confidence)</td>
<td>0.910911</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/1</td>
<td>1082.13 ns [1080.38, 1083.79] (0.95 confidence)</td>
<td>1137.94 ns [1133.55, 1142.31] (0.95 confidence)</td>
<td>0.950954</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/16</td>
<td>1187.4 ns [1058.31, 1425.78] (0.95 confidence)</td>
<td>1129.54 ns [1126.23, 1132.81] (0.95 confidence)</td>
<td>1.05122</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/2</td>
<td>1096.58 ns [1082.33, 1119.99] (0.95 confidence)</td>
<td>1167.49 ns [1159.15, 1176.17] (0.95 confidence)</td>
<td>0.939258</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;100</td>
<td>1274.44 ns [1272.78, 1276.13] (0.95 confidence)</td>
<td>1360.23 ns [1358.54, 1362.01] (0.95 confidence)</td>
<td>0.93693</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/1</td>
<td>1087.42 ns [1085.81, 1088.96] (0.95 confidence)</td>
<td>1123.56 ns [1119.06, 1128.16] (0.95 confidence)</td>
<td>0.967833</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/16</td>
<td>1070.58 ns [1068.78, 1072.49] (0.95 confidence)</td>
<td>1119.3 ns [1115.28, 1123.71] (0.95 confidence)</td>
<td>0.956475</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/2</td>
<td>1072.1 ns [1069.98, 1074.09] (0.95 confidence)</td>
<td>1116.6 ns [1112.6, 1120.59] (0.95 confidence)</td>
<td>0.960151</td>
</tr>
</tbody>
</table>

### Median (ns)

<table>
<thead>
<tr>
<th>Workload</th>
<th>Baseline</th>
<th>Current</th>
<th>Baseline / current</th>
</tr>
</thead>
<tbody>
<tr>
<td>chain/step&#95;by&#95;value</td>
<td>16.626 ns [16.6094, 16.6486] (0.95 confidence)</td>
<td>17.2348 ns [17.226, 17.244] (0.95 confidence)</td>
<td>0.964676</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;accept&#95;reflection</td>
<td>10.4564 ns [10.4293, 10.4753] (0.95 confidence)</td>
<td>10.9828 ns [10.9673, 11.0148] (0.95 confidence)</td>
<td>0.952068</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;no&#95;plan</td>
<td>0.720925 ns [0.720336, 0.721789] (0.95 confidence)</td>
<td>0.728668 ns [0.728212, 0.729466] (0.95 confidence)</td>
<td>0.989375</td>
</tr>
<tr>
<td>chain/step&#95;delayed&#95;reject&#95;reflection</td>
<td>10.6668 ns [10.6277, 10.7003] (0.95 confidence)</td>
<td>11.0572 ns [11.047, 11.0712] (0.95 confidence)</td>
<td>0.964697</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;accept</td>
<td>13.8999 ns [13.8658, 13.9445] (0.95 confidence)</td>
<td>14.2836 ns [14.2559, 14.2997] (0.95 confidence)</td>
<td>0.973132</td>
</tr>
<tr>
<td>chain/step&#95;mut&#95;reject&#95;rollback</td>
<td>192.64 ns [191.988, 193.133] (0.95 confidence)</td>
<td>199.297 ns [198.973, 199.54] (0.95 confidence)</td>
<td>0.966595</td>
</tr>
<tr>
<td>observing/manual&#95;online&#95;sum&#95;100</td>
<td>1262.18 ns [1259.8, 1265.22] (0.95 confidence)</td>
<td>1355.52 ns [1352.02, 1359.11] (0.95 confidence)</td>
<td>0.93114</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;buffer&#95;100</td>
<td>1694.29 ns [1691.47, 1696.46] (0.95 confidence)</td>
<td>1754.33 ns [1751.22, 1755.55] (0.95 confidence)</td>
<td>0.965776</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;binning&#95;100</td>
<td>2282.23 ns [2280.95, 2285.84] (0.95 confidence)</td>
<td>2826.35 ns [2823.98, 2832.11] (0.95 confidence)</td>
<td>0.807483</td>
</tr>
<tr>
<td>observing/run&#95;observing&#95;into&#95;online&#95;stats&#95;100</td>
<td>1440.4 ns [1436.53, 1444.03] (0.95 confidence)</td>
<td>1750.39 ns [1744.61, 1752.61] (0.95 confidence)</td>
<td>0.822902</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;100</td>
<td>1654.77 ns [1652.39, 1657.39] (0.95 confidence)</td>
<td>1771.51 ns [1769.39, 1773.76] (0.95 confidence)</td>
<td>0.934102</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/1</td>
<td>1734.91 ns [1732.46, 1736.66] (0.95 confidence)</td>
<td>1811.48 ns [1805.34, 1816.97] (0.95 confidence)</td>
<td>0.95773</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/16</td>
<td>1730.08 ns [1726.76, 1733.06] (0.95 confidence)</td>
<td>1823.24 ns [1820.32, 1826.83] (0.95 confidence)</td>
<td>0.948904</td>
</tr>
<tr>
<td>sampler/run&#95;by&#95;value&#95;thinned&#95;100/2</td>
<td>1732.56 ns [1729.66, 1735.53] (0.95 confidence)</td>
<td>1833.06 ns [1825.4, 1838.66] (0.95 confidence)</td>
<td>0.945178</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;reflection&#95;100</td>
<td>851.269 ns [849.577, 852.946] (0.95 confidence)</td>
<td>934.769 ns [933.214, 936.377] (0.95 confidence)</td>
<td>0.910674</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/1</td>
<td>1082.5 ns [1080.77, 1084.98] (0.95 confidence)</td>
<td>1146.61 ns [1122.61, 1151.87] (0.95 confidence)</td>
<td>0.944095</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/16</td>
<td>1058.06 ns [1055.87, 1059.55] (0.95 confidence)</td>
<td>1134.65 ns [1122.56, 1138.19] (0.95 confidence)</td>
<td>0.932504</td>
</tr>
<tr>
<td>sampler/run&#95;delayed&#95;thinned&#95;100/2</td>
<td>1076.18 ns [1070.5, 1083.61] (0.95 confidence)</td>
<td>1153.58 ns [1149.91, 1156.94] (0.95 confidence)</td>
<td>0.932909</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;100</td>
<td>1275.22 ns [1272.11, 1276.91] (0.95 confidence)</td>
<td>1360.86 ns [1360.17, 1361.27] (0.95 confidence)</td>
<td>0.937073</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/1</td>
<td>1087.95 ns [1086.5, 1088.98] (0.95 confidence)</td>
<td>1129.36 ns [1110.5, 1133.66] (0.95 confidence)</td>
<td>0.96333</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/16</td>
<td>1071.08 ns [1069.3, 1072.51] (0.95 confidence)</td>
<td>1115.76 ns [1110.94, 1117.45] (0.95 confidence)</td>
<td>0.959953</td>
</tr>
<tr>
<td>sampler/run&#95;mut&#95;thinned&#95;100/2</td>
<td>1072.85 ns [1071.35, 1074.26] (0.95 confidence)</td>
<td>1118.12 ns [1106.02, 1128.45] (0.95 confidence)</td>
<td>0.959509</td>
</tr>
</tbody>
</table>
