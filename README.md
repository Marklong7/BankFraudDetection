# BankFraudDetection

## Key words:
Fraud Detection, Privacy Preserving Machine Learning, Federated Learning, Fairness/bia Analysis.

## Authors:
*Jialong (Mark) Li, Zhitao Zeng*

## The Dataset:
The **Bank Account Fraud (BAF) dataset**, released by Jesus et al. [1] at **NeurIPS 2022**, is a large-scale, privacy-preserving suite of realistic tabular datasets. It includes 1M observations and 30 features, including sensitive features such as age group and employment status. Differential Privacy and necessary Categorization are applied to ensure the dataset is both privacy-protected and realistic. Furthermore, the authors use the CTGAN model to enrich the dataset to 2.5M instances and sample 1M of them to reduce the data size.

Derived from a major consumer bank, the BAF dataset was used to detect fraudulent online bank account applications. In this scenario, fraudsters attempt to access banking services with fake information, then quickly max out credit lines or use the accounts to receive illicit payments.

## Reference:
[1] Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R., ... & Bizarro, P. (2022). Turning the tables: Biased, imbalanced, dynamic tabular datasets for ml evaluation. Advances in Neural Information Processing Systems, 35, 33563-33575.
