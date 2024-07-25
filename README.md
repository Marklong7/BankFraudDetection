# BankFraudDetection

## Key Words:
Fraud Detection, Privacy Preserving Machine Learning, Federated Learning, Fairness/bia Analysis.

## The Dataset:
The **Bank Account Fraud (BAF) dataset**, released by Jesus et al. [1] at NeurIPS 2022, is a large-scale, privacy-preserving suite of realistic tabular datasets. Feature Selection, Differential Privacy, and necessary Categorization are applied to ensure the dataset is both privacy-protected and realistic. Based on the noised dataset, the authors then use the CTGAN model to generate 2.5M instances and sample 1M of them to make the observed month distribution aligned to the original dataset. There are 30 features in the dataset, including sensitive features such as age and income.

Derived from a major consumer bank, the BAF dataset was used to detect fraudulent online bank account applications. In this scenario, fraudsters attempt to access banking services with fake information, then quickly max out credit lines or use the accounts to receive illicit payments.

### Performance Metric:
We select the threshold in order to obtain 5% false positive rate (FPR), and measure the true positive rate (TPR) at that point.

### Fairness Metric:
We measure the ratio between FPRs, i.e., predictive equality [2]. The ratio is calculated by dividing the FPR of the group with lowest observed FPR with the FPR of the group with the highest FPR.

## Experiment & Models:
Building on the framework established by the KDD paper [3], which utilized Logistic Regression (LR), XGBoost (XGB), Random Forest (RF), and Feed Forward Neural Network (MLP), we excluded Decision Tree (as it usually perform poorly) and incorporated more advanced deep learning models.

Additionally, we advanced beyond their research by implementing statistical techniques such as normalization [4], transformation, resampling, and data augmentation (VAE and GAN). These methodologies have significantly enhanced both model performance and fairness, as evidenced in our comprehensive report **[report-benchmark]**. Moreover, generative models like VAE and GAN offer the added benefit of better protecting the privacy of original data.

Yotam Elor's large-scale experiments on 73 datasets [5] demonstrated that balancing could improve prediction performance for weak classifiers but not for the SOTA classifiers (lightGBM, XGBoost, Catboost). However, our preliminary experiments found that using balanced data can significantly improve the fairness score of the model,  which is particularly important in the financial industry.

Furthermore, we conducted in-depth analyses focusing on feature importance and causal inference, yielding interpretable machine learning insights. For more details, refer to **[report-analysis]**.

Finally, we trained federated learning versions of these models. In our previous non-federated learning setup, we only used one-third of the available data for training. However, in the federated learning training setting, we assume that we are collaborating with three companies. They shared their data to train a federated learning model, thereby granting us full access to the entire dataset. This collaborative approach provided us with access to a larger and more diverse dataset than previously possible, demonstrating the scalability and data-sharing advantages of federated learning in distributed settings. We will compare the performance of the two methods, check **[report-federated-learning]** for more detail.

## Contributors:
*Jialong (Mark) Li, Zhitao Zeng, Shuting (Shannon) Fang, Yuhong Shao*

## Acknowledgement:
Illinois Risk Lab.

## Reference:
> [1] Jesus, Sérgio, et al. "Turning the tables: Biased, imbalanced, dynamic tabular datasets for ml evaluation." Advances in Neural Information Processing Systems 35 (2022): 33563-33575.

> [2] Corbett-Davies, Sam, et al. "Algorithmic decision making and the cost of fairness." Proceedings of the 23rd acm sigkdd international conference on knowledge discovery and data mining. 2017.

> [3] Pombal, J., et al. Understanding Unfairness in Fraud Detection through Model and Data Bias Interactions. KDD Workshop on Machine Learning in Finance, August 14–18, 2022, Washington DC

> [4] Du, Z., et al. (2022, December). Rethinking normalization methods in federated learning. In Proceedings of the 3rd International Workshop on Distributed Machine Learning (pp. 16-22).

> [5] Elor, Y., & Averbuch-Elor, H. (2022). To SMOTE, or not to SMOTE?. arXiv preprint arXiv:2201.08528.

> [6] Dong, P., et al. (2024). Privacy-Enhancing Collaborative Information Sharing through Federated Learning--A Case of the Insurance Industry. arXiv preprint arXiv:2402.14983.

