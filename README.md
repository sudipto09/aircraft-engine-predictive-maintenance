#  Overview
Aircraft engine failures are safety-critical and extremely costly.
This project builds a leak-free machine learning pipeline to predict the Remaining Useful Life (RUL) of turbofan aircraft engines using multivariate sensor data.

The goal is to estimate how many operational cycles remain before engine failure, enabling predictive maintenance instead of reactive servicing.

#
Predictive maintenance is widely used by:

Aircraft manufacturers (e.g. engine OEMs)

Airlines and MROs (Maintenance, Repair, Overhaul)

Safety-critical monitoring systems

Accurate RUL prediction helps:

Reduce unscheduled maintenance

Improve fleet availability

Enhance operational safety

Lower lifecycle costs

# Dataset
NASA CMAPSS Turbofan Engine Dataset (FD001)

->Simulated degradation of turbofan engines

->Multivariate time-series sensor data

->Each engine runs until failure

->Target: Remaining Useful Life (RUL)

Important preprocessing detail:

The CMAPSS data files are whitespace-delimited, not single-space delimited.

Correct parsing is done using:

pd.read_csv(..., delim_whitespace=True)


Incorrect parsing leads to silent data corruption and invalid results.

# Key Challenges
This project explicitly handles several common pitfalls in predictive maintenance:

Engine-wise data leakage
→ Validation engines are completely unseen during training

Silent target leakage detection
→ Dummy and linear models used as sanity checks

Missing sensor values
→ Mean imputation fitted only on training data

Time-series structure
→ Rolling window features 

#  Methodology
Data Preparation:

Correct CMAPSS parsing (delim_whitespace=True)

RUL computation from final engine cycle

RUL capping (optional, industry standard)

Validation Strategy:

Engine-wise train/validation split

No overlap of engines between sets

Prevents temporal and entity leakage

Feature Selection:

Selected informative sensors based on CMAPSS literature

Removed constant / non-informative signals

Missing Data Handling:

Mean imputation using SimpleImputer

Imputer fitted only on training data

Models:

Dummy Regressor (sanity check)

Linear Regression (baseline)

Random Forest Regressor (nonlinear model)

# Results
| Model                            | RMSE (cycles)       |
| -------------------------------- | ------------------- |
| Dummy Regressor                  | ~43                 |
| Linear Regression                | ~21                 |
| Random Forest                    | ~18                 |

Random Forest outperforms linear models by capturing nonlinear degradation patterns in sensor data.

# Visualization

Predicted vs True RUL plots are saved in:
results/figures/

These plots help visually assess:

Bias near end-of-life

Prediction spread

Model reliability

# Tech Stack
Python

Pandas, NumPy

scikit-learn

Matplotlib

Jupyter Notebook

# Learning Outcomes

Correct handling of time-series entity data

Detection and prevention of silent data leakage

Industry-grade validation practices

Practical aerospace ML pipeline design
