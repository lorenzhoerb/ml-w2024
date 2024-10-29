# Datasets

This is an overview of the assigened/choosen datasets for exercise 1.

## Congressional Voting
- https://www.kaggle.com/competitions/184-702-tu-ml-ws-24-congressional-voting/data


### Goal

The goal is to predict the political party affiliation (Democrat or Republican) of congressional members based on their voting history on various legislative issues.

### Data Overview

The dataset contains voting records of congress members represented by a series of legislative decisions. Each members voting position is categorized by ‘y’ (yes), ‘n’ (no) and ‘unknown’

### Features

- ID: Ratio
- class (democrat, republican): Nominal ← Value to classify (predict)
- handicapped-infants (y, n, unknown): Nominal
- water-project-cost-sharing (y, n, unknown): Nominal
- adoption-of-the-budget-resolution (y, n, unknown): Nominal
- physician-fee-freeze (y, n, unknown): Nominal
- el-salvador-aid (y, n, unknown): Nominal
- religious-groups-in-schools (y, n, unknown): Nominal
- anti-satellite-test-ban (y, n, unknown): Nominal
- aid-to-nicaraguan-contras (y, n, unknown): Nominal
- mx-missile (y, n, unknown): Nominal
- immigration (y, n, unknown): Nominal
- synfuels-crporation-cutback (y, n, unknown): Nominal
- education-spending  (y, n, unknown): Nominal
- superfund-right-to-sue (y, n, unknown): Nominal
- crime (y, n, unknown): Nominal
- duty-free-exports (y, n, unknown): Nominal
- export-administration-act-south-africa (y, n, unknown): Nominal

## Credit Score
https://www.openml.org/search?type=data&status=active&id=31&sort=runs

This dataset classifies people described by a set of attributes as good or bad credit risks. 

It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).

- Number of Instances: 1000
- **Target class: good | bad**

### Features

- **Status of existing checking account:** Nominal (attribute name: `checking_status`)
  - A11: ... < 0 DM
  - A12: 0 <= ... < 200 DM
  - A13: ... >= 200 DM / salary assignments for at least 1 year
  - A14: no checking account

- **Duration:** Numerical (attribute name: `duration`)
  - Duration in months

- **Credit history:** Nominal (attribute name: `credit_history`)
  - A30: no credits taken / all credits paid back duly
  - A31: all credits at this bank paid back duly
  - A32: existing credits paid back duly till now
  - A33: delay in paying off in the past
  - A34: critical account / other credits existing (not at this bank)

- **Purpose:** Nominal (attribute name: `purpose`)
  - A40: car (new)
  - A41: car (used)
  - A42: furniture/equipment
  - A43: radio/television
  - A44: domestic appliances
  - A45: repairs
  - A46: education
  - A47: (vacation - does not exist?)
  - A48: retraining
  - A49: business
  - A410: others

- **Credit amount:** Numerical (attribute name: `credit_amount`)
  - Amount of credit in DM

- **Savings account/bonds:** Nominal (attribute name: `savings_status`)
  - A61: ... < 100 DM
  - A62: 100 <= ... < 500 DM
  - A63: 500 <= ... < 1000 DM
  - A64: ... >= 1000 DM
  - A65: unknown / no savings account

- **Present employment since:** Nominal (attribute name: `employment`)
  - A71: unemployed
  - A72: ... < 1 year
  - A73: 1 <= ... < 4 years
  - A74: 4 <= ... < 7 years
  - A75: ... >= 7 years

- **Installment rate:** Numerical (attribute name: `installment_commitment`)
  - Percentage of disposable income

- **Personal status and sex:** Nominal (attribute name: `personal_status`)
  - A91: male : divorced/separated
  - A92: female : divorced/separated/married
  - A93: male : single
  - A94: male : married/widowed
  - A95: female : single

- **Other debtors / guarantors:** Nominal (attribute name: `other_parties`)
  - A101: none
  - A102: co-applicant
  - A103: guarantor

- **Present residence since:** Numerical (attribute name: `residence_since`)
  - Duration in years

- **Property:** Nominal (attribute name: `property_magnitude`)
  - A121: real estate
  - A122: if not A121: building society savings agreement / life insurance
  - A123: if not A121/A122: car or other, not in attribute 6
  - A124: unknown / no property

- **Age:** Numerical (attribute name: `age`)
  - Age in years

- **Other installment plans:** Nominal (attribute name: `other_payment_plans`)
  - A141: bank
  - A142: stores
  - A143: none

- **Housing:** Nominal (attribute name: `housing`)
  - A151: rent
  - A152: own
  - A153: for free

- **Number of existing credits at this bank:** Numerical (attribute name: `existing_credits`)
  - Total number of credits

- **Job:** Nominal (attribute name: `job`)
  - A171: unemployed / unskilled - non-resident
  - A172: unskilled - resident
  - A173: skilled employee / official
  - A174: management / self-employed / highly qualified employee / officer

- **Number of people being liable to provide maintenance for:** Numerical (attribute name: `num_dependents`)
  - Count of dependents

- **Telephone:** Nominal (attribute name: `own_telephone`)
  - A191: none
  - A192: yes, registered under the customer's name

- **Foreign worker:** Nominal (attribute name: `foreign_worker`)
  - A201: yes
  - A202: no
