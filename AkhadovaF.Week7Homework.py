import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Core Activity files one by one
act_2010 = pd.read_csv('Downloads/atusact_2010/atusact_2010.dat')
act_2012 = pd.read_csv('Downloads/atusact_2012/atusact_2012.dat')
act_2013 = pd.read_csv('Downloads/atusact_2013/atusact_2013.dat')

# Stack them on top of each other
act_df = pd.concat([act_2010, act_2012, act_2013])

# 2. Load the Well-Being file (this one is already pooled for all years)
wb_act = pd.read_csv('Downloads/wbact_1013/wbact_1013.dat')

# 3. Merge Happiness data with Activity data
# This is like matching a puzzle: it connects "How I felt" to "What I did"
df = pd.merge(wb_act, act_df, on=['TUCASEID', 'TUACTIVITY_N'], how='inner')

# 4. Load the "Who" files one by one (to see if they were alone)
who_2010 = pd.read_csv('Downloads/atuswho_2010/atuswho_2010.dat')
who_2012 = pd.read_csv('Downloads/atuswho_2012/atuswho_2012.dat')
who_2013 = pd.read_csv('Downloads/atuswho_2013/atuswho_2013.dat')

# Stack the Who files
who_df = pd.concat([who_2010, who_2012, who_2013])

# Create a simple True/False column: Is the person alone? (Code 18 = Alone)
who_df['is_alone'] = (who_df['TUWHO_CODE'] == 18)

# Group the "Who" data so we have one answer per activity
# This simplifies the data so we can merge it easily
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

#Filter out other stuff so that there's only leisure 
analysis_df = final_data[final_data['Activity_Group'] != 'Other']


analysis_df.columns = analysis_df.columns.str.strip().str.lower()

#Results
print("\nAverage Happiness Scores:")
print(analysis_df.groupby(['activity_group', 'is_alone'])['wuhappy'].mean())

### Visualization
# 1. Change True/False to words
analysis_df['social_context'] = analysis_df['is_alone'].replace({True: 'Alone', False: 'With Others'})

# 2. Make the Bar Chart
# y='wuhappy'
sns.barplot(data=analysis_df, x='activity_group', y='wuhappy', hue='social_context')

plt.title('Average Happiness: Screen vs. Non-Screen')
plt.xlabel('Activity Type')
plt.ylabel('Average Happiness (0-6)')

plt.show()