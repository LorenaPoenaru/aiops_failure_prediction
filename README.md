# McUDI: Model-Centric Unsupervised Degradation Indicator for Failure Prediction AIOps Solutions



### Project structure:
```
.
├── datasets                    include the two datasets, google_job_failure.csv and disk_failure_2015.csv to run the scripts
├── results                     Results (as .csvs  and plots) get saved to this folder
├── scripts                     Scripts used to get our results from the paper. Scripts are both Jupyter Notebooks and R files.

```

### Extracting the Drift Ground Truth:
First, run the Jupyter Notebooks _disk_failure_extract_errors_for_ground_truth.ipynb_ and _job_failure_extract_errors_for_ground_truth.ipynb_ . The scripts will output the .csv files from *results* _concept_drift_disk_2015_rf_week_feature_importance_rs1.csv_ and _concept_drift_job_rf_feature_importance_rs1.csv_ respectively. Then run the 2 R scripts _concept_drift_batches_extraction_disk.R_ and _concept_drift_batches_extraction_job.R_, respectively. The resulting .csv files from the R scripts are stored in the *results* folder (_rf_concept_drift_localization_disk_week_2015_r_1.csv_ and _rf_concept_drift_localization_job_r_1.csv_). In this package, we give an example of using the random seed 1234. Check the scripts for the other random seeds we experimented with.

### RQ1
RQ1 is answered using scripts that include rq1 in the script title. 

### RQ2
RQ2 is answered using scripts that include rq2 in the script title. 

### RQ3
RQ3 is answered using scripts that include rq3 in the script title. 

### Requirements
Python==3.11.1
numpy==1.25.1
pandas==2.0.2
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.12.2
matplotlib==3.7.2
tqdm==4.65.0
