## 1. the report-benchmark
### Logistics only
* For linear classifier, log-transformation can improve both model performance and model fairness.
* Newton's method yields better performance than stochastic optimization methods in this case.
### Main Insights
* None Deep Learnings models usually do not achieve good fairness scores if no balancing techniques is applied.
* Re-balancing techniques can improve the model fairness score significantly, with little cost of model performance loss or even enhance the model perforcement - always use them when you have an imblanced dataset.
* Compared to re-sampling method (random over sampling, random under sampling, SMOTE, SMOTESVM, VAE), use balanced class weight always has decent performance and fairness score.
* VAE outperforms 
