we propose Track-Tennis, an enhanced TrackNet framework with three complementary improvements. A lightweight GhostNet encoder reduces the number of model parameters by 49.8%, providing computational capacity for a spatio-temporal transformer that explicitly captures long-range spatial context and inter-frame temporal dependencies. A bidirectional pyramid fusion decoder further aggregates multi-scale features to improve the localization of fast-moving small objects. Experiments on the TennisTrack dataset demonstrate that Track-Tennis achieves 94.79% accuracy with a 4.18% miss rate, outperforming the original TrackNet by 1.21 percentage points while reducing the parameter count by 39.9%.
<img width="641" height="389" alt="image" src="https://github.com/user-attachments/assets/4d205eeb-7888-4f79-8b8d-5ede9b546cf6" />
Model	Parameters (M)	FLOPs (G)	Reduction vs. Baseline (Params)
TrackNet_Baseline	11.34	113.80	-
TrackNet_Ghost	5.69	57.40	49.80%
TrackNet_GhostAtt	5.72	58.01	49.60%
Track-tennis	6.81	159.55	39.90%
Model	TP	TN	FP1	FP2	FN	Acc	Prec	Recall	F1	MissRate
Baseline	1444	28	7	2	92	0.9358	0.9938	0.9401	0.9662	0.0599
Ghost	1437	28	15	2	91	0.9313	0.9883	0.9404	0.9638	0.0596
Ghost+Att	1456	28	7	2	80	0.9434	0.9939	0.9479	0.9703	0.0521
Track-tennis	1468	23	11	7	64	0.9479	0.9879	0.9582	0.9728	0.0418
