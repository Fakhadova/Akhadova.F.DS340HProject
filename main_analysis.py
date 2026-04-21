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
nonscreen_codes = [120312, 120309, 120301] # Reading, Hobbies, Relaxing

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

#Check how many activity episodes each respondent contributes
rows_per_person = analysis_df.groupby('tucaseid').size()

print("\nHow many respondents contribute how many rows?")
print(rows_per_person.value_counts().sort_index())

print("\nSummary of rows per respondent:")
print(rows_per_person.describe())

#How many respondents have more than 1 row?
num_repeat_people = (rows_per_person > 1).sum()
print("\nNumber of respondents with more than 1 row:", num_repeat_people)

#How many rows come from respondents with more than 1 row?
repeat_ids = rows_per_person[rows_per_person > 1].index
rows_from_repeat_people = analysis_df['tucaseid'].isin(repeat_ids).sum()
print("Number of rows from respondents with more than 1 row:", rows_from_repeat_people)

#What proportion of all rows are from repeated respondents?
print("Proportion of rows from repeated respondents:",
      rows_from_repeat_people / len(analysis_df))

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
        'weighted_happiness': weighted_mean(x['wuhappy'], x['wufnactwtp']),
        'weighted_stress': weighted_mean(x['wustress'], x['wufnactwtp'])
    })
).reset_index()

print("\nWeighted results table:")
print(weighted_table)

#Save the weighted table in case I want to include it in my homework write-up
weighted_table.to_csv("/Users/farangizakhadova/Downloads/weighted_results_table.csv", index=False)

#Non-screen leisure done with others has the highest weighted average happiness, at 4.785,
#screen-based leisure done alone has the lowest, at 4.063. 
#The stress differences are smaller overall, but the same group pattern appears 
#non-screen leisure with others has the lowest weighted stress, 0.969, and screen-based leisure alone has the highest, 1.079.

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
    weights=analysis_df['wufnactwtp']
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
    weights=analysis_df['wufnactwtp']
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

#happiness pattern still looks more meaningful than the stress pattern. 
#In the happiness model, being with others is associated with a 0.554-point increase in happiness than being alone
#That result was statistically significant
#The interaction between screen-based leisure and being with others is -0.266 and also statistically significant (p = .020)
#in the stress model, the interaction is very small (0.029) and not statistically significant (p = .776)

#new numbers are 4.090 and 4.785


import statsmodels.api as sm


# MIXED-EFFECTS MODELS
# Random intercept for respondent ID

# If re_formula is left out, the default is a random intercept.

# For MixedLM, use a clean copy with the needed variables only
mixed_df = analysis_df[['tucaseid', 'wuhappy', 'wustress',
                        'activity_group', 'social_context',
                        'teage', 'tesex', 'telfs']].dropna().copy()

# Make sure the categorical variables are treated as categories
mixed_df['activity_group'] = mixed_df['activity_group'].astype('category')
mixed_df['social_context'] = mixed_df['social_context'].astype('category')
mixed_df['tesex'] = mixed_df['tesex'].astype('category')
mixed_df['telfs'] = mixed_df['telfs'].astype('category')

# Mixed Model 1: Happiness
mixed_happy_model = sm.MixedLM.from_formula(
    'wuhappy ~ C(activity_group) * C(social_context) + teage + C(tesex) + C(telfs)',
    groups='tucaseid',
    data=mixed_df
)

mixed_happy_result = mixed_happy_model.fit(reml=False, method='lbfgs')

print("\nMIXED MODEL RESULTS: HAPPINESS")
print(mixed_happy_result.summary())

# Mixed Model 2: Stress
mixed_stress_model = sm.MixedLM.from_formula(
    'wustress ~ C(activity_group) * C(social_context) + teage + C(tesex) + C(telfs)',
    groups='tucaseid',
    data=mixed_df
)

mixed_stress_result = mixed_stress_model.fit(reml=False, method='lbfgs')

print("\nMIXED MODEL RESULTS: STRESS")
print(mixed_stress_result.summary())

# Save mixed model summaries
with open("/Users/farangizakhadova/Downloads/mixed_happiness_model_summary.txt", "w") as f:
    f.write(mixed_happy_result.summary().as_text())

with open("/Users/farangizakhadova/Downloads/mixed_stress_model_summary.txt", "w") as f:
    f.write(mixed_stress_result.summary().as_text())

    #Create a comparison table of the most important coefficients
comparison_table = pd.DataFrame({
    'Result to compare': [
        'Happiness: Screen-Based',
        'Happiness: With Others',
        'Happiness interaction',
        'Stress: Screen-Based',
        'Stress: With Others',
        'Stress interaction',
        'Respondent random intercept variance'
    ],
    'WLS + clustered SE': [
        f"{happiness_model.params['C(activity_group)[T.Screen-Based]']:.3f} (p = {happiness_model.pvalues['C(activity_group)[T.Screen-Based]']:.3f})",
        f"{happiness_model.params['C(social_context)[T.With Others]']:.3f} (p < .001)" if happiness_model.pvalues['C(social_context)[T.With Others]'] < 0.001 else f"{happiness_model.params['C(social_context)[T.With Others]']:.3f} (p = {happiness_model.pvalues['C(social_context)[T.With Others]']:.3f})",
        f"{happiness_model.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f} (p = {happiness_model.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f})",
        f"{stress_model.params['C(activity_group)[T.Screen-Based]']:.3f} (p = {stress_model.pvalues['C(activity_group)[T.Screen-Based]']:.3f})",
        f"{stress_model.params['C(social_context)[T.With Others]']:.3f} (p = {stress_model.pvalues['C(social_context)[T.With Others]']:.3f})",
        f"{stress_model.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f} (p = {stress_model.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f})",
        "—"
    ],
    'Mixed-effects model': [
        f"{mixed_happy_result.params['C(activity_group)[T.Screen-Based]']:.3f} (p = {mixed_happy_result.pvalues['C(activity_group)[T.Screen-Based]']:.3f})",
        f"{mixed_happy_result.params['C(social_context)[T.With Others]']:.3f} (p < .001)" if mixed_happy_result.pvalues['C(social_context)[T.With Others]'] < 0.001 else f"{mixed_happy_result.params['C(social_context)[T.With Others]']:.3f} (p = {mixed_happy_result.pvalues['C(social_context)[T.With Others]']:.3f})",
        f"{mixed_happy_result.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f} (p = {mixed_happy_result.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f})",
        f"{mixed_stress_result.params['C(activity_group)[T.Screen-Based]']:.3f} (p = {mixed_stress_result.pvalues['C(activity_group)[T.Screen-Based]']:.3f})",
        f"{mixed_stress_result.params['C(social_context)[T.With Others]']:.3f} (p = {mixed_stress_result.pvalues['C(social_context)[T.With Others]']:.3f})",
        f"{mixed_stress_result.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f} (p = {mixed_stress_result.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']:.3f})",
        f"{mixed_happy_result.params['tucaseid Var']:.3f} (happiness), {mixed_stress_result.params['tucaseid Var']:.3f} (stress)"
    ]
})

print("\nCOMPARISON TABLE")
print(comparison_table)

#Save it in case you want to use it in the write-up
comparison_table.to_csv("/Users/farangizakhadova/Downloads/model_comparison_table.csv", index=False)

# ONE GRAPHIC: point-range plot with two side-by-side panels
# Put this after weighted_table is created

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate weighted mean
def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

# Function to calculate approximate weighted standard error
def weighted_se(values, weights):
    mean = weighted_mean(values, weights)

    # weighted variance
    weighted_var = np.sum(weights * (values - mean) ** 2) / np.sum(weights)

    # effective sample size
    n_eff = (np.sum(weights) ** 2) / np.sum(weights ** 2)

    # standard error
    se = np.sqrt(weighted_var / n_eff)
    return se

# Create table for plotting
plot_table = analysis_df.groupby(['activity_group', 'social_context']).apply(
    lambda x: pd.Series({
        'happiness_mean': weighted_mean(x['wuhappy'], x['wufnactwtp']),
        'happiness_se': weighted_se(x['wuhappy'], x['wufnactwtp']),
        'stress_mean': weighted_mean(x['wustress'], x['wufnactwtp']),
        'stress_se': weighted_se(x['wustress'], x['wufnactwtp'])
    })
).reset_index()

# Set order so groups appear nicely
group_order = [
    ('Non-Screen', 'Alone'),
    ('Non-Screen', 'With Others'),
    ('Screen-Based', 'Alone'),
    ('Screen-Based', 'With Others')
]

plot_table['sort_key'] = plot_table.apply(
    lambda row: group_order.index((row['activity_group'], row['social_context'])),
    axis=1
)

plot_table = plot_table.sort_values('sort_key').reset_index(drop=True)

# Create combined x-axis labels
plot_table['group_label'] = [
    'Non-Screen\nAlone',
    'Non-Screen\nWith Others',
    'Screen-Based\nAlone',
    'Screen-Based\nWith Others'
]

# X positions
x = np.arange(len(plot_table))

# Create one figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)

# Left panel: Happiness
axes[0].errorbar(
    x,
    plot_table['happiness_mean'],
    yerr=1.96 * plot_table['happiness_se'],
    fmt='o',
    capsize=5
)
axes[0].set_title('Happiness')
axes[0].set_ylabel('Weighted Mean (0–6)')
axes[0].set_xlabel('Group')
axes[0].set_xticks(x)
axes[0].set_xticklabels(plot_table['group_label'])
axes[0].set_ylim(0, 6)

# Right panel: Stress
axes[1].errorbar(
    x,
    plot_table['stress_mean'],
    yerr=1.96 * plot_table['stress_se'],
    fmt='o',
    capsize=5
)
axes[1].set_title('Stress')
axes[1].set_ylabel('Weighted Mean (0–6)')
axes[1].set_xlabel('Group')
axes[1].set_xticks(x)
axes[1].set_xticklabels(plot_table['group_label'])
axes[1].set_ylim(0, 6)

# Overall title
fig.suptitle('Weighted Mean Well-Being by Activity Type and Social Context', fontsize=14)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# MODEL SUMMARY OUTPUTS FOR GOOGLE SHEETS
# Put this AFTER all four models are fit
# --------------------------------------------------

import pandas as pd
import numpy as np

# ---------- helper functions ----------

def format_p(p):
    if p < 0.001:
        return "< .001"
    else:
        return f"{p:.3f}"

def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# Pseudo-R^2 for mixed models:
# marginal R^2 = variance explained by fixed effects / total variance
# conditional R^2 = variance explained by fixed + random effects / total variance
def mixed_r2(result, df, outcome_name):
    # fixed-effects fitted values
    fixed_fitted = result.predict(df)
    var_fixed = np.var(fixed_fitted, ddof=1)

    # residual variance
    var_resid = result.scale

    # random intercept variance
    # cov_re is usually a 1x1 matrix for random intercept model
    var_random = float(result.cov_re.iloc[0, 0])

    total_var = var_fixed + var_random + var_resid

    marginal_r2 = var_fixed / total_var
    conditional_r2 = (var_fixed + var_random) / total_var

    return marginal_r2, conditional_r2

# ---------- get mixed-model pseudo-R^2 ----------
mixed_happy_marginal_r2, mixed_happy_conditional_r2 = mixed_r2(
    mixed_happy_result, mixed_df, 'wuhappy'
)

mixed_stress_marginal_r2, mixed_stress_conditional_r2 = mixed_r2(
    mixed_stress_result, mixed_df, 'wustress'
)

# ---------- fit statistics table ----------
fit_stats = pd.DataFrame({
    'model': ['WLS', 'WLS', 'Mixed-effects', 'Mixed-effects'],
    'outcome': ['Happiness', 'Stress', 'Happiness', 'Stress'],
    'aic': [
        happiness_model.aic,
        stress_model.aic,
        mixed_happy_result.aic,
        mixed_stress_result.aic
    ],
    'bic': [
        happiness_model.bic,
        stress_model.bic,
        mixed_happy_result.bic,
        mixed_stress_result.bic
    ],
    'log_likelihood': [
        happiness_model.llf,
        stress_model.llf,
        mixed_happy_result.llf,
        mixed_stress_result.llf
    ],
    'r_squared': [
        happiness_model.rsquared,
        stress_model.rsquared,
        mixed_happy_marginal_r2,
        mixed_stress_marginal_r2
    ],
    'adj_r_squared_or_conditional_r2': [
        happiness_model.rsquared_adj,
        stress_model.rsquared_adj,
        mixed_happy_conditional_r2,
        mixed_stress_conditional_r2
    ]
})

print("\nFIT STATISTICS")
print(fit_stats)

fit_stats.to_csv("/Users/farangizakhadova/Downloads/model_fit_stats_for_sheets.csv", index=False)

# ---------- main coefficient table ----------
results_table = pd.DataFrame({
    'result': [
        'Screen-Based (Happiness)',
        'With Others (Happiness)',
        'Screen-Based × With Others (Happiness)',
        'Screen-Based (Stress)',
        'With Others (Stress)',
        'Screen-Based × With Others (Stress)'
    ],
    'wls_coef': [
        happiness_model.params['C(activity_group)[T.Screen-Based]'],
        happiness_model.params['C(social_context)[T.With Others]'],
        happiness_model.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        stress_model.params['C(activity_group)[T.Screen-Based]'],
        stress_model.params['C(social_context)[T.With Others]'],
        stress_model.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ],
    'wls_p': [
        happiness_model.pvalues['C(activity_group)[T.Screen-Based]'],
        happiness_model.pvalues['C(social_context)[T.With Others]'],
        happiness_model.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        stress_model.pvalues['C(activity_group)[T.Screen-Based]'],
        stress_model.pvalues['C(social_context)[T.With Others]'],
        stress_model.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ],
    'wls_sig': [
        stars(happiness_model.pvalues['C(activity_group)[T.Screen-Based]']),
        stars(happiness_model.pvalues['C(social_context)[T.With Others]']),
        stars(happiness_model.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']),
        stars(stress_model.pvalues['C(activity_group)[T.Screen-Based]']),
        stars(stress_model.pvalues['C(social_context)[T.With Others]']),
        stars(stress_model.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'])
    ],
    'mixed_coef': [
        mixed_happy_result.params['C(activity_group)[T.Screen-Based]'],
        mixed_happy_result.params['C(social_context)[T.With Others]'],
        mixed_happy_result.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        mixed_stress_result.params['C(activity_group)[T.Screen-Based]'],
        mixed_stress_result.params['C(social_context)[T.With Others]'],
        mixed_stress_result.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ],
    'mixed_p': [
        mixed_happy_result.pvalues['C(activity_group)[T.Screen-Based]'],
        mixed_happy_result.pvalues['C(social_context)[T.With Others]'],
        mixed_happy_result.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        mixed_stress_result.pvalues['C(activity_group)[T.Screen-Based]'],
        mixed_stress_result.pvalues['C(social_context)[T.With Others]'],
        mixed_stress_result.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ],
    'mixed_sig': [
        stars(mixed_happy_result.pvalues['C(activity_group)[T.Screen-Based]']),
        stars(mixed_happy_result.pvalues['C(social_context)[T.With Others]']),
        stars(mixed_happy_result.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']),
        stars(mixed_stress_result.pvalues['C(activity_group)[T.Screen-Based]']),
        stars(mixed_stress_result.pvalues['C(social_context)[T.With Others]']),
        stars(mixed_stress_result.pvalues['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'])
    ]
})

print("\nMAIN RESULTS TABLE")
print(results_table)

results_table.to_csv("/Users/farangizakhadova/Downloads/model_results_for_sheets.csv", index=False)

# ---------- optional: nicely formatted text version ----------
formatted_table = pd.DataFrame({
    'Result': results_table['result'],
    'WLS + clustered SE': [
        f"{coef:.3f} (p {format_p(p)}) {sig}"
        for coef, p, sig in zip(results_table['wls_coef'], results_table['wls_p'], results_table['wls_sig'])
    ],
    'Mixed-effects model': [
        f"{coef:.3f} (p {format_p(p)}) {sig}"
        for coef, p, sig in zip(results_table['mixed_coef'], results_table['mixed_p'], results_table['mixed_sig'])
    ]
})

print("\nFORMATTED TABLE")
print(formatted_table)

formatted_table.to_csv("/Users/farangizakhadova/Downloads/model_results_formatted_for_sheets.csv", index=False)


print(happiness_model.aic)
print(stress_model.aic)
print(mixed_happy_result.aic)
print(mixed_stress_result.aic)

print(happiness_model.bic)
print(stress_model.bic)
print(mixed_happy_result.bic)

print(happiness_model.llf)
print(stress_model.llf)
print(mixed_happy_result.llf)
print(mixed_stress_result.llf)

print(happiness_model.rsquared)
print(stress_model.rsquared)
print(mixed_happy_marginal_r2)
print(mixed_stress_marginal_r2)
print(happiness_model.rsquared_adj)
print(stress_model.rsquared_adj)
print(mixed_happy_conditional_r2)
print(mixed_stress_conditional_r2)

import pandas as pd
import matplotlib.pyplot as plt

# Create age groups
analysis_df['age_group'] = pd.cut(
    analysis_df['teage'],
    bins=[15, 24, 34, 44, 54, 64, 74, 85],
    labels=['15-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-85'],
    include_lowest=True
)

age_counts = analysis_df['age_group'].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(age_counts.index.astype(str), age_counts.values)
plt.title('Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Number of Episodes')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Build table of coefficients and 95% CIs
coef_data = pd.DataFrame({
    'term': [
        'Screen-Based (Happiness)',
        'With Others (Happiness)',
        'Screen-Based × With Others (Happiness)',
        'Screen-Based (Stress)',
        'With Others (Stress)',
        'Screen-Based × With Others (Stress)'
    ],
    'wls_coef': [
        happiness_model.params['C(activity_group)[T.Screen-Based]'],
        happiness_model.params['C(social_context)[T.With Others]'],
        happiness_model.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        stress_model.params['C(activity_group)[T.Screen-Based]'],
        stress_model.params['C(social_context)[T.With Others]'],
        stress_model.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ],
    'wls_se': [
        happiness_model.bse['C(activity_group)[T.Screen-Based]'],
        happiness_model.bse['C(social_context)[T.With Others]'],
        happiness_model.bse['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        stress_model.bse['C(activity_group)[T.Screen-Based]'],
        stress_model.bse['C(social_context)[T.With Others]'],
        stress_model.bse['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ],
    'mixed_coef': [
        mixed_happy_result.params['C(activity_group)[T.Screen-Based]'],
        mixed_happy_result.params['C(social_context)[T.With Others]'],
        mixed_happy_result.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        mixed_stress_result.params['C(activity_group)[T.Screen-Based]'],
        mixed_stress_result.params['C(social_context)[T.With Others]'],
        mixed_stress_result.params['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ],
    'mixed_se': [
        mixed_happy_result.bse['C(activity_group)[T.Screen-Based]'],
        mixed_happy_result.bse['C(social_context)[T.With Others]'],
        mixed_happy_result.bse['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]'],
        mixed_stress_result.bse['C(activity_group)[T.Screen-Based]'],
        mixed_stress_result.bse['C(social_context)[T.With Others]'],
        mixed_stress_result.bse['C(activity_group)[T.Screen-Based]:C(social_context)[T.With Others]']
    ]
})

# Plot positions
y = np.arange(len(coef_data))

plt.figure(figsize=(10, 6))

# WLS points and CIs
plt.errorbar(
    coef_data['wls_coef'],
    y + 0.12,
    xerr=1.96 * coef_data['wls_se'],
    fmt='o',
    capsize=4,
    label='WLS + clustered SE'
)

# Mixed model points and CIs
plt.errorbar(
    coef_data['mixed_coef'],
    y - 0.12,
    xerr=1.96 * coef_data['mixed_se'],
    fmt='o',
    capsize=4,
    label='Mixed-effects'
)

plt.axvline(0, linestyle='--')
plt.yticks(y, coef_data['term'])
plt.xlabel('Coefficient Estimate')
plt.title('Main Coefficients from WLS and Mixed-Effects Models')
plt.legend()
plt.tight_layout()
plt.show()
# Count episodes in each group
group_counts = analysis_df.groupby(['activity_group', 'social_context']).size().reset_index(name='n')

# Convert to strings before combining
group_counts['group_label'] = (
    group_counts['activity_group'].astype(str) + "\n" +
    group_counts['social_context'].astype(str)
)

# Order the groups
order = [
    'Non-Screen\nAlone',
    'Non-Screen\nWith Others',
    'Screen-Based\nAlone',
    'Screen-Based\nWith Others'
]

group_counts['group_label'] = pd.Categorical(
    group_counts['group_label'],
    categories=order,
    ordered=True
)

group_counts = group_counts.sort_values('group_label')

plt.figure(figsize=(8,5))

plt.hlines(
    y=group_counts['group_label'],
    xmin=0,
    xmax=group_counts['n']
)

plt.plot(group_counts['n'], group_counts['group_label'], 'o')

for _, row in group_counts.iterrows():
    plt.text(row['n'] + 60, row['group_label'], f"{row['n']:,}", va='center')

plt.title('Distribution of Episodes Across the Four Comparison Groups')
plt.xlabel('Number of Episodes')
plt.ylabel('')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# ----------------------------
# WEIGHTED MEAN FUNCTIONS
# ----------------------------
def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

def weighted_se(values, weights):
    mean = weighted_mean(values, weights)

    # weighted variance
    weighted_var = np.sum(weights * (values - mean) ** 2) / np.sum(weights)

    # effective sample size
    n_eff = (np.sum(weights) ** 2) / np.sum(weights ** 2)

    # standard error
    se = np.sqrt(weighted_var / n_eff)
    return se


# ----------------------------
# WEIGHTED SUMMARY TABLE
# ----------------------------
weighted_table = analysis_df.groupby(['activity_group', 'social_context']).apply(
    lambda x: pd.Series({
        'n_episodes': len(x),
        'weighted_happiness': weighted_mean(x['wuhappy'], x['wufnactwtp']),
        'weighted_stress': weighted_mean(x['wustress'], x['wufnactwtp'])
    })
).reset_index()

print("\nWeighted results table:")
print(weighted_table)

weighted_table.to_csv("/Users/farangizakhadova/Downloads/weighted_results_table.csv", index=False)


# ----------------------------
# INTERACTIVE BAR CHART
# thinner bars
# ----------------------------
fig = px.bar(
    weighted_table,
    x='activity_group',
    y='weighted_happiness',
    color='social_context',
    barmode='group',
    title='Weighted Happiness: Screen-Based vs. Non-Screen Leisure',
    labels={
        'weighted_happiness': 'Happiness Score',
        'activity_group': 'Activity Type',
        'social_context': 'Context'
    }
)

# make bars thinner
fig.update_traces(width=0.28)

# a little more spacing for cleaner look
fig.update_layout(
    bargap=0.45,
    bargroupgap=0.18
)

fig.write_html("/Users/farangizakhadova/Downloads/weighted_happiness_chart.html")
fig.show()


# ----------------------------
# STATIC BAR CHARTS
# thinner bars
# ----------------------------
bar_order = ['Non-Screen', 'Screen-Based']
hue_order = ['Alone', 'With Others']

# Plot 1: Weighted Happiness
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=weighted_table,
    x='activity_group',
    y='weighted_happiness',
    hue='social_context',
    order=bar_order,
    hue_order=hue_order,
    ci=None
)

# manually make bars thinner
new_width = 0.28
for patch in ax.patches:
    current_width = patch.get_width()
    diff = current_width - new_width
    patch.set_width(new_width)
    patch.set_x(patch.get_x() + diff / 2)

plt.title('Weighted Happiness by Activity Type and Social Context')
plt.ylabel('Weighted Mean Happiness (0–6)')
plt.xlabel('Activity Type')
plt.ylim(0, 5.2)
plt.legend(title='Social Context')
plt.tight_layout()
plt.show()

# Plot 2: Weighted Stress
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=weighted_table,
    x='activity_group',
    y='weighted_stress',
    hue='social_context',
    order=bar_order,
    hue_order=hue_order,
    ci=None
)

# manually make bars thinner
new_width = 0.28
for patch in ax.patches:
    current_width = patch.get_width()
    diff = current_width - new_width
    patch.set_width(new_width)
    patch.set_x(patch.get_x() + diff / 2)

plt.title('Weighted Stress by Activity Type and Social Context')
plt.ylabel('Weighted Mean Stress (0–6)')
plt.xlabel('Activity Type')
plt.ylim(0, 1.3)
plt.legend(title='Social Context')
plt.tight_layout()
plt.show()

# ----------------------------
# POINT-RANGE PLOT
# horizontal version with less white space
# ----------------------------
plot_table = analysis_df.groupby(['activity_group', 'social_context']).apply(
    lambda x: pd.Series({
        'happiness_mean': weighted_mean(x['wuhappy'], x['wufnactwtp']),
        'happiness_se': weighted_se(x['wuhappy'], x['wufnactwtp']),
        'stress_mean': weighted_mean(x['wustress'], x['wufnactwtp']),
        'stress_se': weighted_se(x['wustress'], x['wufnactwtp'])
    })
).reset_index()

group_order = [
    ('Non-Screen', 'Alone'),
    ('Non-Screen', 'With Others'),
    ('Screen-Based', 'Alone'),
    ('Screen-Based', 'With Others')
]

plot_table['sort_key'] = plot_table.apply(
    lambda row: group_order.index((row['activity_group'], row['social_context'])),
    axis=1
)

plot_table = plot_table.sort_values('sort_key').reset_index(drop=True)

plot_table['group_label'] = [
    'Non-Screen, Alone',
    'Non-Screen, With Others',
    'Screen-Based, Alone',
    'Screen-Based, With Others'
]

plot_table['happy_ci'] = 1.96 * plot_table['happiness_se']
plot_table['stress_ci'] = 1.96 * plot_table['stress_se']

y = np.arange(len(plot_table))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

# Happiness panel
axes[0].errorbar(
    plot_table['happiness_mean'],
    y,
    xerr=plot_table['happy_ci'],
    fmt='o',
    capsize=4
)
axes[0].set_title('Happiness')
axes[0].set_xlabel('Weighted Mean (0–6)')
axes[0].set_yticks(y)
axes[0].set_yticklabels(plot_table['group_label'])
axes[0].invert_yaxis()

happy_min = (plot_table['happiness_mean'] - plot_table['happy_ci']).min()
happy_max = (plot_table['happiness_mean'] + plot_table['happy_ci']).max()
axes[0].set_xlim(happy_min - 0.12, happy_max + 0.18)

for i, row in plot_table.iterrows():
    label_x = row['happiness_mean'] + row['happy_ci'] + 0.02
    axes[0].text(label_x, i, f"{row['happiness_mean']:.2f}", va='center')

# Stress panel
axes[1].errorbar(
    plot_table['stress_mean'],
    y,
    xerr=plot_table['stress_ci'],
    fmt='o',
    capsize=4
)
axes[1].set_title('Stress')
axes[1].set_xlabel('Weighted Mean (0–6)')

stress_min = (plot_table['stress_mean'] - plot_table['stress_ci']).min()
stress_max = (plot_table['stress_mean'] + plot_table['stress_ci']).max()
axes[1].set_xlim(stress_min - 0.08, stress_max + 0.12)

for i, row in plot_table.iterrows():
    label_x = row['stress_mean'] + row['stress_ci'] + 0.01
    axes[1].text(label_x, i, f"{row['stress_mean']:.2f}", va='center')

fig.suptitle('Weighted Mean Well-Being by Activity Type and Social Context', fontsize=14)
plt.tight_layout()
plt.show()

print("Happiness WLS R-squared:", round(happiness_model.rsquared, 4))
print("Happiness WLS Adjusted R-squared:", round(happiness_model.rsquared_adj, 4))
print("Difference:", round(happiness_model.rsquared - happiness_model.rsquared_adj, 4))

print("Stress WLS R-squared:", round(stress_model.rsquared, 4))
print("Stress WLS Adjusted R-squared:", round(stress_model.rsquared_adj, 4))
print("Difference:", round(stress_model.rsquared - stress_model.rsquared_adj, 4))
