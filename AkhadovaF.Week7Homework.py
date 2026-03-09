import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

#Filter out other stuff so that there's only leisure (missing values)
analysis_df = final_data[final_data['Activity_Group'] != 'Other']
analysis_df.columns = analysis_df.columns.str.strip().str.lower()

#New: Filter out missing happiness and stress values (-2, -3)
analysis_df = analysis_df[(analysis_df['wuhappy'] >= 0) & (analysis_df['wustress'] >= 0)]       

#Visualization #1
# 1. Create the social_context column using the lowercase 'is_alone'
analysis_df['social_context'] = analysis_df['is_alone'].replace({True: 'Alone', False: 'With Others'})

# 2. Group data
plot_data = analysis_df.groupby(['activity_group', 'social_context'])['wuhappy'].mean().reset_index()

# 3. Create the interactive bar chart
fig = px.bar(
    plot_data, 
    x='activity_group', 
    y='wuhappy', 
    color='social_context',
    barmode='group',
    title='Average Happiness: Screen-Based vs. Non-Screen Leisure',
    labels={'wuhappy': 'Happiness Score', 'activity_group': 'Activity Type', 'social_context': 'Context'}
)

# 4. Save interactive plot as HTML
fig.write_html("/Users/farangizakhadova/Downloads/happiness_map_style.html")

# 5. Show
fig.show()

#Weighted means calculation
#Creating weighted averages to represent the US population
def get_weighted_stats(data):
    #Calculate the weighted mean for happiness using wufnactwt
    h_mean = (data['wuhappy'] * data['wufnactwt']).sum()/data['wufnactwt'].sum()
    #Calculate the weighted mean for stress using wufnactwt
    s_mean = (data['wustress'] * data['wufnactwt']).sum()/data['wufnactwt'].sum()
    return pd.Series({'weighted_happiness': h_mean, 'weighted_stress': s_mean})

#Create the summary table for plotting
plot_data = analysis_df.groupby(['activity_group', 'social_context']).apply(get_weighted_stats).reset_index()

#Visualizations #2
#Plot 1: Weighted Happiness
plt.figure(figsize=(10, 6))
sns.barplot(x='activity_group', y='weighted_happiness', hue='social_context', data=plot_data)
plt.title('Weighted Happiness by Activity and Context')
plt.ylabel('Weighted Happiness Score (0-6)')
plt.ylim(0,5)
plt.show()

#Plot #2: Weighted Stress
plt.figure(figsize=(10, 6))
sns.barplot(x='activity_group', y='weighted_stress', hue='social_context', data=plot_data)
plt.title('Weighted Stress by Activity and Context')
plt.ylabel('Weighted Stress Score (0-6)')
plt.ylim(0,5)
plt.show()
