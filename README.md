# BankFraudDetection

## Key words:
Fraud Detection, Privacy Preserving Machine Learning, Federated Learning, Fairness/bia Analysis.

## Authors:
*Jialong (Mark) Li, Zhitao Zeng*

## The Dataset:
The **Bank Account Fraud (BAF) dataset**, released by Jesus et al. [1] at **NeurIPS 2022**, is a large-scale, privacy-preserving suite of realistic tabular datasets. It includes 1M observations and 30 features, including sensitive features such as age and income. Differential Privacy and necessary Categorization are applied to ensure the dataset is both privacy-protected and realistic. Furthermore, the authors use the CTGAN model to enrich the dataset to 2.5M instances and sample 1M of them to reduce the data size.

Derived from a major consumer bank, the BAF dataset was used to detect fraudulent online bank account applications. In this scenario, fraudsters attempt to access banking services with fake information, then quickly max out credit lines or use the accounts to receive illicit payments.

* Performance Metric:
We select the threshold in order to obtain 5% false positive rate (FPR), and measure the true positive rate (TPR) at that point.

* Fairness Metric:
We measure the ratio between FPRs, i.e., predictive equality [2]. The ratio is calculated by dividing the FPR of the group with lowest observed FPR with the FPR of the group with the highest FPR.

## Reference:
[1] Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R., ... & Bizarro, P. (2022). Turning the tables: Biased, imbalanced, dynamic tabular datasets for ml evaluation. Advances in Neural Information Processing Systems, 35, 33563-33575.
[2] Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq. Algorithmic Decision Making and the Cost of Fairness. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’17, pages 797–806, New York, NY, USA, August 2017. Association for Computing Machinery
