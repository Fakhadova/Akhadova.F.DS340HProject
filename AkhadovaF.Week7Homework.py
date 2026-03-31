import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import statsmodels.formula.api as smf


# 1. Load Data
#Loading Activity Files
act_df = pd.concat ([pd.read_csv('/Users/farangizakhadova/Downloads/atusact_2010/atusact_2010.dat'),
                     pd.read_csv('/Users/farangizakhadova/Downloads/atusact_2012/atusact_2012.dat'),
                     pd.read_csv('/Users/farangizakhadova/Downloads/atusact_2013/atusact_2013.dat')])

# Load the Well-Being file (this one is already pooled for all years)
wb_act = pd.read_csv('/Users/farangizakhadova/Downloads/wbact_1013/wbact_1013.dat')

#Loading Respondent data for Age (TEAGE) and Employment (TELFS)
resp_df = pd.concat ([pd.read_csv('/Users/farangizakhadova/Downloads/atusresp_2010/atusresp_2010.dat'),
                        pd.read_csv('/Users/farangizakhadova/Downloads/atusresp_2012/atusresp_2012.dat'),
                        pd.read_csv('/Users/farangizakhadova/Downloads/atusresp_2013/atusresp_2013.dat')])


roster_df = pd.concat([pd.read_csv('/Users/farangizakhadova/Downloads/atusrost_2010/atusrost_2010.dat'),
                     pd.read_csv('/Users/farangizakhadova/Downloads/atusrost_2012/atusrost_2012.dat'),
                     pd.read_csv('/Users/farangizakhadova/Downloads/atusrost_2013/atusrost_2013.dat')])

roster_df = roster_df[roster_df['TULINENO'] == 1] # Keep only the main respondent (TULINENO = 1)


#Merging
# 1. First, create the 'df' by merging happiness and activity info
df = pd.merge(wb_act, act_df, on=['TUCASEID', 'TUACTIVITY_N'], how='inner')

# 2. Clean Roster column names (to ensure TEAGE/TESEX are found)
roster_df.columns = roster_df.columns.str.strip().str.upper()

# 3. Merge with Roster (for Age/Sex)
df = pd.merge(df, roster_df[['TUCASEID', 'TEAGE', 'TESEX']], on='TUCASEID', how='left')

# 4. Clean Respondent column names (for TELFS)
resp_df.columns = resp_df.columns.str.strip().str.upper()

# 5. Merge with Respondent (for Employment Status)
df = pd.merge(df, resp_df[['TUCASEID', 'TELFS']], on='TUCASEID', how='left')
#Alone Status and Categories
who_df = pd.concat([pd.read_csv('/Users/farangizakhadova/Downloads/atuswho_2010/atuswho_2010.dat'),
                    pd.read_csv('/Users/farangizakhadova/Downloads/atuswho_2012/atuswho_2012.dat'),
                    pd.read_csv('/Users/farangizakhadova/Downloads/atuswho_2013/atuswho_2013.dat')])
# Create a simple True/False column: Is the person alone? (Code 18 = Alone)
who_df['is_alone'] = (who_df['TUWHO_CODE'] == 18)
alone_status = who_df.groupby(['TUCASEID', 'TUACTIVITY_N'])['is_alone'].all().reset_index()
# 5. Final Join: Put it all together into one big table
final_data = pd.merge(df, alone_status, on=['TUCASEID', 'TUACTIVITY_N'], how='left')

# 6. Check first few rows
print(final_data.head())


#Create a 6-digit activity code for easy filtering 
#This combines the three tier columns into one 
# Tier 1 moves to the front, Tier 2 to the middle, Tier 3 stays at the end
final_data['activity_6d'] = (final_data['TUTIER1CODE'] * 10000) + \
                             (final_data['TUTIER2CODE'] * 100) + \
                             (final_data['TUTIER3CODE'])

#Check the distribution of activities
activity_counts = final_data['activity_6d'].value_counts()
print(activity_counts.head())

#Define the gruops based on the ATUS Lexicon
#Exactly which chodes are "Screen" and "non-screen"
screen_codes = [120303, 120308, 120307] # TV, Computer Leisure, Games
nonscreen_codes = [120312, 120309, 120101, 120301] # Reading, Hobbies, Socializing, Relaxing

#Create a new column that labels everything as "Other" first
final_data['Activity_Group'] = 'Other'
#Use isin() to label two main groups. 
final_data.loc[final_data['activity_6d'].isin(screen_codes), 'Activity_Group'] = 'Screen-Based'
final_data.loc[final_data['activity_6d'].isin(nonscreen_codes), 'Activity_Group'] = 'Non-Screen'

#check missing values in main variables before filtering
print(final_data[['TEAGE', 'TELFS', 'is_alone', 'WUHAPPY']].isnull().sum())



#Filter out other stuff so that there's only leisure (missing values)
analysis_df = final_data[final_data['Activity_Group'] != 'Other']

#make all column names lowercase
analysis_df.columns = analysis_df.columns.str.strip().str.lower()

#New: Filter out missing happiness and stress values (-2, -3)
analysis_df = analysis_df[(analysis_df['wuhappy'] >= 0) & (analysis_df['wustress'] >= 0)]      

# Remove any rows where the 'is_alone' merge failed
analysis_df = analysis_df.dropna(subset=['is_alone'])

print(analysis_df['activity_group'].value_counts())
# Safety filter: Ensure Age and Employment also have no negative survey codes
analysis_df = analysis_df[analysis_df['teage'] >= 0]
analysis_df = analysis_df[analysis_df['telfs'] >= 0]

print(f"Final clean sample size: {len(analysis_df)} rows")

#create social context lables 
#turns the true/false variable into nicer labels for tables and plots
analysis_df['social_context'] = analysis_df['is_alone'].replace({
    True: 'Alone',
    False: 'With Others'
})

#project progress (3/30) - Descriptive tables + weighted results
#sample description 

#Sample Description
#Count how many unique respondents and how many total activity episodes are in the final data
num_respondents = analysis_df['tucaseid'].nunique()
num_episodes = len(analysis_df)

print("Number of respondents:", num_respondents)
print("Number of activity episodes:", num_episodes)

#Show how many episodes are in each leisure type
print("\nEpisodes by activity group:")
print(analysis_df['activity_group'].value_counts())

#Show the four main comparison groups
print("\nEpisodes by activity group and social context:")
group_counts = analysis_df.groupby(['activity_group', 'social_context']).size().reset_index(name='n')
print(group_counts)

#Basic sample characteristics
print("\nSummary of age:")
print(analysis_df['teage'].describe())

print("\nSex distribution:")
print(analysis_df['tesex'].value_counts(dropna=False))

print("\nEmployment status distribution:")
print(analysis_df['telfs'].value_counts(dropna=False))

#Weighted means calculation
#Creating weighted averages to represent the US population
def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum()

#Create a weighted summary table for the four groups
weighted_table = analysis_df.groupby(['activity_group', 'social_context']).apply(
    lambda x: pd.Series({
        'n_episodes': len(x),
        'weighted_happiness': weighted_mean(x['wuhappy'], x['wufnactwt']),
        'weighted_stress': weighted_mean(x['wustress'], x['wufnactwt'])
    })
).reset_index()

print("\nWeighted results table:")
print(weighted_table)

#Save the weighted table in case I want to include it in my homework write-up
weighted_table.to_csv("/Users/farangizakhadova/Downloads/weighted_results_table.csv", index=False)

#Visualization #1
#Interactive bar chart for weighted happiness
fig = px.bar(
    weighted_table,
    x='activity_group',
    y='weighted_happiness',
    color='social_context',
    barmode='group',
    title='Weighted Happiness: Screen-Based vs. Non-Screen Leisure',
    labels={'weighted_happiness': 'Happiness Score', 'activity_group': 'Activity Type', 'social_context': 'Context'}
)

#Save interactive plot as HTML
fig.write_html("/Users/farangizakhadova/Downloads/weighted_happiness_chart.html")

#Show plot
fig.show()

#Visualizations #2
#Plot 1: Weighted Happiness
plt.figure(figsize=(10, 6))
sns.barplot(x='activity_group', y='weighted_happiness', hue='social_context', data=weighted_table)
plt.title('Weighted Happiness by Activity and Context')
plt.ylabel('Weighted Happiness Score (0-6)')
plt.xlabel('Activity Type')
plt.ylim(0, 6)
plt.tight_layout()
plt.show()

#Plot #2: Weighted Stress
plt.figure(figsize=(10, 6))
sns.barplot(x='activity_group', y='weighted_stress', hue='social_context', data=weighted_table)
plt.title('Weighted Stress by Activity and Context')
plt.ylabel('Weighted Stress Score (0-6)')
plt.xlabel('Activity Type')
plt.ylim(0, 6)
plt.tight_layout()
plt.show()

#Prepare variables for regression
#Convert these to category so statsmodels treats them as groups instead of regular numbers
analysis_df['activity_group'] = analysis_df['activity_group'].astype('category')
analysis_df['social_context'] = analysis_df['social_context'].astype('category')
analysis_df['tesex'] = analysis_df['tesex'].astype('category')
analysis_df['telfs'] = analysis_df['telfs'].astype('category')

#Run Model 1: Happiness
#This tests whether happiness differs by leisure type, social context,
#and the interaction between the two, while controlling for age, sex, and employment
happiness_model = smf.wls(
    formula='wuhappy ~ C(activity_group) * C(social_context) + teage + C(tesex) + C(telfs)',
    data=analysis_df,
    weights=analysis_df['wufnactwt']
).fit(
    cov_type='cluster',
    cov_kwds={'groups': analysis_df['tucaseid']}
)

print("\nHAPPINESS MODEL RESULTS")
print(happiness_model.summary())

#Run Model 2: Stress
#This does the same thing, but now stress is the outcome
stress_model = smf.wls(
    formula='wustress ~ C(activity_group) * C(social_context) + teage + C(tesex) + C(telfs)',
    data=analysis_df,
    weights=analysis_df['wufnactwt']
).fit(
    cov_type='cluster',
    cov_kwds={'groups': analysis_df['tucaseid']}
)

print("\nSTRESS MODEL RESULTS")
print(stress_model.summary())

#Save model summaries as text files
#This makes it easier to copy results into the write-up later
with open("/Users/farangizakhadova/Downloads/happiness_model_summary.txt", "w") as f:
    f.write(happiness_model.summary().as_text())

with open("/Users/farangizakhadova/Downloads/stress_model_summary.txt", "w") as f:
    f.write(stress_model.summary().as_text())