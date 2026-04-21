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
- `Poster Draft 2 - Farangiz Akhadova.pdf` — most recent poster draft
- `main_analysis.py` — main Python analysis file
- `README.md` — project overview and repository guide
- `data/` — raw ATUS files used in the analysis and a small sample of the cleaned analytic dataset

## Data Folder
The `data/` folder contains:
- activity files for 2010, 2012, and 2013
- Well-Being Module activity data
- respondent files for 2010, 2012, and 2013
- roster files for 2010, 2012, and 2013
- Who files for 2010, 2012, and 2013
- a small sample of the cleaned analytic dataset

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
Download the required ATUS files and update the file paths in the Python script if needed. Then run:

`main_analysis.py`

The script produces descriptive tables, model output, and figures used in the poster.
