import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import pandas_profiling as pp
from scipy.stats import norm
import seaborn as sns
import os

pd.set_option('display.max_columns', None)


for dirname, _, filenames in os.walk('./dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


folder_loc = './dataset/' #changes 
quality_data_loc = 'quality_attributes' #changes
#files = glob.glob(folder_loc+'*.csv')
files = []
#fixed_folders = ['/kaggle/input/sqa-dataset/quality_attributes/1 spring-framework', '/kaggle/input/sqa-dataset/quality_attributes/2 junit-5' ]
for dirname, _, filenames in os.walk(folder_loc+quality_data_loc):
    #print(dirname)
    #if dirname in fixed_folders:
    #print (str(len(files))  + dirname)
    for filename in filenames:
            #if(filename == '2020-7.csv'):#latest version
        files.append(os.path.join(dirname, filename))

#files.pop(0)
#files = files[3:] # remove first 3 elments 
files
data = pd.concat([pd.read_csv(fp).assign(filename=os.path.basename(fp).split('.')[0], projectname=os.path.dirname(fp).split('/')[-1]    ) for fp in files])


#--------------------------------

professional_repo = ['1 spring-framework', '2 junit-5', '3 kafka-trunk', '4 lucene-solr-master', '8 selenium-trunk']
open_source_repo = ['5 dropwizard-master', '6 checkstyle-master', '7 hadoop-trunk', '9 skywalking-master', '10 Signal-Android-master']

low_size_repo = [ '2 junit-5', '3 kafka-trunk', '5 dropwizard-master', '6 checkstyle-master', '8 selenium-trunk']
high_size_repo = ['1 spring-framework',  '4 lucene-solr-master',  '7 hadoop-trunk', '9 skywalking-master', '10 Signal-Android-master']

low_age_repo = [ '2 junit-5', '3 kafka-trunk' , '5 dropwizard-master',  '9 skywalking-master', '10 Signal-Android-master' ]
high_age_repo = ['1 spring-framework', '4 lucene-solr-master', '6 checkstyle-master', '7 hadoop-trunk', '8 selenium-trunk']

# create a list of our conditions
condition_type1 = [
    (data['projectname'].isin(professional_repo)),
    (data['projectname'].isin(open_source_repo))
    ]

condition_type2 = [
    (data['projectname'].isin(low_size_repo)),
    (data['projectname'].isin(high_size_repo))
    ]

condition_type3 = [
    (data['projectname'].isin(low_age_repo)),
    (data['projectname'].isin(high_age_repo))
    ]

# create a list of the values we want to assign for each condition
values_type_1 = ['Professional', 'Open-source']
values_type_2 = ['Low Volume Repository', 'High Volume Repository']
values_type_3 = ['Age < 10 Years', 'Age > 10 Years']

# create a new column and use np.select to assign values to it using our lists as arguments
data['Type 1'] = np.select(condition_type1, values_type_1)
data['Type 2'] = np.select(condition_type2, values_type_2)
data['Type 3'] = np.select(condition_type3, values_type_3)

# display updated DataFrame
dataset_detail = pd.read_csv(folder_loc+'attribute-details.csv')
dataset_detail.head(50)


def codes_detail(codes, desc_show=False):
    idxs = dataset_detail.index[ dataset_detail['Code'].isin(codes) ]
    print ('---------------------------------------')
    i=1
    for idx in idxs:
        print (str(i) + '. '+ dataset_detail.at[idx,'Code']+ '\t: '+ dataset_detail.at[idx,'Full name'])
        if desc_show:
            print(' - '+ dataset_detail.at[idx,'Description'])
        i = i + 1
    print ('---------------------------------------')
def code_detail(code):
    idx = dataset_detail.index[ dataset_detail['Code'] == code ]
    print ('---------------------------------------')
    print ('Code: '+ dataset_detail.at[idx[0],'Code'])
    print ('---------------------------------------')
    print ('Category: '+ dataset_detail.at[idx[0],'Category'])
    print ('Short Name: '+ dataset_detail.at[idx[0],'Short name'])
    print ('Full Name: '+ dataset_detail.at[idx[0],'Full name'])
    print ('Description: '+ dataset_detail.at[idx[0],'Description'])
    print ('---------------------------------------')
    
code_detail('CBO')
codes_detail(['LOC.2', 'LOC'])

class_attributes = dataset_detail[ dataset_detail['Category'].str.contains ('Class')==True ]
class_attributes.head(100)


data_classes = data[~data['QualifiedName'].str.contains("<Package>|<Method>|<Field>")]
print(class_attributes['Code'].tolist())
data_classes = data_classes[['QualifiedName', 'Name', 'Type 1', 'Type 2','Type 3', 'Coupling', 'Lack of Cohesion', 'Complexity', 'Size', 'LOC', 'WMC', 'DIT', 'NOC', 'CBO', 'RFC', 'SRFC', 'LCOM', 'LCAM', 'NOF', 'NOM', 'NOSF', 'NOSM', 'SI', 'CMLOC', 'NORM', 'LTCC', 'ATFD', 'filename', 'projectname'  ]]
#classes_row = classes_row[class_attributes['Code'].tolist()]
data_classes = data_classes.dropna()


# change_values(df, 'Complexity', 'low', 'lw')
def change_values(df, column_name, from_str, to_str):
    row = (df[column_name].str.contains(from_str))
    df.loc[row, column_name] = to_str
    return df

data_classes = change_values(data_classes, 'Complexity', 'low', 'low')    
data_classes = change_values(data_classes, 'Complexity', 'high', 'high') 

data_classes = change_values(data_classes, 'Coupling', 'low', 'low')    
data_classes = change_values(data_classes, 'Coupling', 'high', 'high') 

data_classes = change_values(data_classes, 'Size', 'low', 'low')    
data_classes = change_values(data_classes, 'Size', 'high', 'high') 

data_classes = change_values(data_classes, 'Lack of Cohesion', 'low', 'low')    
data_classes = change_values(data_classes, 'Lack of Cohesion', 'high', 'high') 

print(data_classes.describe())

prof_data = data.loc[ data['projectname'].isin(open_source_repo) ]
prof_classes_row  = data_classes[data_classes['Type 1']== "Professional"]
open_classes_row  = data_classes[data_classes['Type 1'] == 'Open-source']
open_classes_row 

year_order = ['2016-1', '2017-1', '2018-1', '2019-1', '2020-1', '2021-1']

sns_plot = sns.countplot(x="filename", hue="Type 1", data=data_classes, order=year_order)
plt.title("Number of Classes in Years (2016-2021)")
plt.xlabel("Development Years")
plt.ylabel("Class Count")
sns_plot.figure.savefig("number_classes_per_year.png")

sns.kdeplot(data=data_classes, x="DIT", hue="Type 1", multiple="stack")
#sns.kdeplot(data=data_classes, x="LCOM", hue="Type 1", multiple="stack")
#sns.kdeplot(data=data_classes, x="LCAM", hue="Type 1", multiple="stack")

sns.displot(data_classes, x="NOM", hue="Type 1", bins=50, multiple="dodge")
sns.relplot(x="LOC", y="WMC", hue="Type 1", style="filename", data=data_classes);
#only categorical value
vn = 'Size'

sns_plot = sns.catplot(x="filename", hue=vn, col="Type 1", data=data_classes, kind="count", order=year_order);
sns_plot.set_axis_labels("Development Years", "Class Count")
sns_plot.set_titles(vn+" classes in Professional Repo.", vn+" classes in Open-Source Repo.")
sns_plot.savefig("number_"+vn+"_classes_per_year.png")
g = sns.FacetGrid(data_classes, col="Type 1")
g.map(sns.histplot, "filename")

from scipy import stats
def quantile_plot(x, **kwargs):
    quantiles, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, quantiles, **kwargs)

g = sns.FacetGrid(data_classes, col="Type 1", height=4)
g.map(quantile_plot, "ATFD")

def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)

    g = sns.FacetGrid(data_classes, hue="Type 1", col="filename", height=4, col_order=year_order)
#g.map(qqplot, "LOC", "ATFD")
g.map(qqplot, "LOC", "RFC")
g.add_legend()

g = sns.FacetGrid(data_classes, col="Complexity", hue="Type 1")
g.map(sns.scatterplot, "NOSF", "NOSM", alpha=.7)
g.add_legend()

g = sns.lmplot(x="LOC", y="WMC", hue="Type 1", data=data_classes, col="filename", col_order=year_order)

vn = 'LOC' # LOC(best), WMC(good), NOF(taking time), {DIT, NOM, ATFD(same as class com), } 
sns_plot = sns.barplot(x="filename", y=vn, hue="Type 1", data=data_classes, order = year_order)
plt.title(vn+" evolution in Years (2016-2021)")
plt.xlabel("Development Years")
plt.ylabel(vn+" Value")
sns_plot.figure.savefig(vn+"_per_year.png")


vn = 'Complexity' #Complexity ok
sns_plot = open_classes_row[vn].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Low vs High "+vn+" comparison in Open Source Repositories")
sns_plot.figure.savefig(vn+"_pie_open.png")

sns_plot = prof_classes_row['Complexity'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Low vs High Complexity comparison in Professional Repositories")
sns_plot.figure.savefig("complexity_pie_prof.png")