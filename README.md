# Screen-Based vs. Non-Screen Leisure: How Social Context Shapes Momentary Happiness and Stress

## Project Description
This project uses the American Time Use Survey (ATUS) Well-Being Module from 2010, 2012, and 2013 to examine how momentary happiness and stress differ between screen-based and non-screen leisure activities, and whether those patterns differ depending on whether respondents are alone or with others.

## Research Question
How do momentary happiness and stress differ between screen-based and non-screen leisure activities, and do those patterns change depending on whether people are alone or with others?

## Data Sources
This project uses pooled ATUS data from 2010, 2012, and 2013, including:
- activity files
- Well-Being Module activity files
- respondent files
- roster files
- Who files

Source: U.S. Bureau of Labor Statistics, American Time Use Survey (ATUS).

## Repository Contents
- `Final Poster - Farangiz Akhadova.pdf` - final poster
- `main_analysis.py` - main Python analysis file
- `README.md` - project overview and repository guide
- `atus_analysis_sample.csv` - small sample of the cleaned analytic dataset

## Data Access
The full ATUS raw files used in this project were too large to upload to GitHub. This repository includes a small sample of the cleaned analytic dataset to show the project structure and variables used in the analysis.

The analysis code uses ATUS activity, Well-Being Module activity, respondent, roster, and Who files from 2010, 2012, and 2013. These files can be downloaded from the U.S. Bureau of Labor Statistics ATUS website and then used with the script by updating the file paths if needed.

## Methods Summary
- merged ATUS activity, well-being, respondent, roster, and Who files
- created 6-digit activity codes
- grouped selected activities into screen-based and non-screen leisure
- defined social context as alone versus with others
- calculated weighted mean happiness and stress
- estimated weighted least squares models with clustered standard errors
- estimated mixed-effects models with respondent random intercepts

## Main Findings
- happiness differences were clearer than stress differences
- non-screen leisure with others had the highest weighted happiness
- screen-based leisure was associated with lower happiness in the mixed-effects model
- stress results were weaker and less consistent across models

## How to Run
1. Download the ATUS files used in the analysis.
2. Update file paths in `main_analysis.py` if needed.
3. Run `main_analysis.py`.

The script produces descriptive tables, model results, and figures used in the poster.

## Author
Farangiz Akhadova  
Wellesley College  
Data Science Major Capstone
