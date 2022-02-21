# TOP-MAMA
TOP MAMA PROJECT
Automatic saving failed. This file was updated remotely or in another tab. Show diff
TOP UP MAMA
TOP UP MAMA_
[185]
0s
import statsmodels
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

[1]
13m
#Import dataset
#Import file from desktop
import io
from google.colab import files
df = files.upload()
#from google.colab import drive
#drive.mount('/content/drive')


TASK1:1. DATA MERGING INTO ONE FILE

[5]
2s
##Data merging into one file
#combine all 6 csv files into one data frame df
extension='csv'
df = [i for i in glob.glob('*.{}'.format(extension))]
df = pd.concat([pd.read_csv(f) for f in df])
df

DATA CLEANING

[6]
0s
#data types look out for object and float64 as they may contain missing values
df.dtypes
Task_ID                       float64
Order_ID                       object
Relationship                  float64
Team_Name                      object
Task_Type                      object
                               ...   
Number of employees           float64
Upload restuarant location     object
Agent_Name                     object
Unnamed: 34                    object
Unnamed: 35                    object
Length: 90, dtype: object
[7]
0s
#shape of the combined dataframe
df.shape
(71945, 90)
[8]
0s
# View columns names pf the combined dataframe
df.columns
Index(['Task_ID', 'Order_ID', 'Relationship', 'Team_Name', 'Task_Type',
       'Notes', 'Agent_ID', 'Distance(m)', 'Total_Time_Taken(min)',
       'Pick_up_From', 'Start_Before', 'Complete_Before', 'Completion_Time',
       'Task_Status', 'Ref_Images', 'Rating', 'Review', 'Latitude',
       'Longitude', 'Tags', 'Promo_Applied', 'Custom_Template_ID',
       'Task_Details_QTY', 'Task_Details_AMOUNT', 'Special_Instructions',
       'Tip', 'Delivery_Charges', 'Discount', 'Subtotal', 'Payment_Type',
       'Task_Category', 'Earning', 'Pricing', 'Order ID', 'Order Status',
       'Category Name', 'SKU', 'Customization Group', 'Customization Option',
       'Quantity', 'Unit Price', 'Cost Price', 'Total Cost Price',
       'Total Price', 'Order Total', 'Sub Total', 'Tax', 'Delivery Charge',
       'Remaining Balance', 'Payment Method', 'Additional Charge',
       'Taxable Amount', 'Transaction ID', 'Currency Symbol',
       'Transaction Status', 'Promo Code', 'Customer ID', 'Merchant ID',
       'Store Name', 'Pickup Address', 'Description', 'Distance (in km)',
       'Order Time', 'Pickup Time', 'Delivery Time', 'Ratings', 'Reviews',
       'Merchant Earning', 'Commission Amount', 'Commission Payout Status',
       'Order Preparation Time', 'Debt Amount', 'Redeemed Loyalty Points',
       'Consumed Loyalty Points', 'Cancellation Reason', 'Flat Discount',
       'Checkout Template Name', 'Checkout Template Value',
       'Last Used Platform', 'Is Blocked', 'Created At', 'Language',
       'Outstanding Amount', 'Loyalty Points', 'Number of Employees',
       'Number of employees', 'Upload restuarant location', 'Agent_Name',
       'Unnamed: 34', 'Unnamed: 35'],
      dtype='object')
[9]
0s
#We may run a simple statistical description to understand the central tendencies of the combined data set.
# Simple Statistical Description
df.describe()

We can see that 'Longitudes','unit price','Distance' and 'loyalty points' have outliers because the larger the standard deviation to mean, the more spread the data,which may indicate some outliers in those attributes

2.2 Dealing with Missing Values

A Pandas DataFrame, encodes missing values as Null (i.e. NaN). So, we can identify missing values by search all the NaN data points

[10]
0s
#1. Check if the entire data set has any Null values

# Check if the entire data set has any Null values
df.isnull().values.any()

#isnull().values.any()* returns **True** for this data set, meaning we have missing values.
True
isnull().values.any()* returns True for this data set, meaning we have missing values.
[11]
1s
#2. Check which columns/attributes have Null values

# Check which columns have Null values
#df.isna().any()
# Check if any missing or NaN values
df.info

<bound method DataFrame.info of           Task_ID       Order_ID  ...  Unnamed: 34 Unnamed: 35
0     368032956.0  YR-11262518,0  ...          NaN         NaN
1     368032956.0  YR-11262518,0  ...          NaN         NaN
2     368012178.0  YR-11261796,0  ...          NaN         NaN
3     368012178.0  YR-11261796,0  ...          NaN         NaN
4     367999205.0  YR-11261341,0  ...          NaN         NaN
...           ...            ...  ...          ...         ...
1424          NaN            NaN  ...          NaN         NaN
1425          NaN            NaN  ...          NaN         NaN
1426          NaN            NaN  ...          NaN         NaN
1427          NaN            NaN  ...          NaN         NaN
1428          NaN            NaN  ...          NaN         NaN

[71945 rows x 90 columns]>
We observe that only columns 'Task_ID','Order_ID','Relationship','Team_Name','Task_Type','Number of employees','Upload restuarant location','Agent_Name','Unnamed: 34' and 'Unnamed: 35' have Null values.

STEP 1:DATA CLEANING DELETE COLUMNS CONTAINING 70% OR MORE THAN 70% NaN VALUES

[12]
0s
# Missing values
df.isna()

[13]
0s
#sum of missing values
#df.isna().sum()
df.isnull().sum()
Task_ID                       18943
Order_ID                      18943
Relationship                  18943
Team_Name                     18943
Task_Type                     18943
                              ...  
Number of employees           68806
Upload restuarant location    71926
Agent_Name                    26962
Unnamed: 34                   71836
Unnamed: 35                   71906
Length: 90, dtype: int64
[14]
0s
#Percentage of column missing values
df.isna().mean() * 100
Task_ID                       26.329835
Order_ID                      26.329835
Relationship                  26.329835
Team_Name                     26.329835
Task_Type                     26.329835
                                ...    
Number of employees           95.636945
Upload restuarant location    99.973591
Agent_Name                    37.475850
Unnamed: 34                   99.848495
Unnamed: 35                   99.945792
Length: 90, dtype: float64
[15]
0s
# Delete columns containing either 70% or more than 70% NaN Values
perc = 70.0
min_count =  int(((100-perc)/100)*df.shape[0] + 1)
df = df.dropna( axis=1, 
                thresh=min_count)
print("Modified Dataframe : ")
print(df)
Modified Dataframe : 
          Task_ID       Order_ID  Relationship  ... Earning Pricing Agent_Name
0     368032956.0  YR-11262518,0  3.680330e+29  ...       -       -        NaN
1     368032956.0  YR-11262518,0  3.680330e+29  ...       -       -        NaN
2     368012178.0  YR-11261796,0  3.680122e+29  ...       -       -        NaN
3     368012178.0  YR-11261796,0  3.680122e+29  ...       -       -        NaN
4     367999205.0  YR-11261341,0  3.679992e+29  ...       -       -        NaN
...           ...            ...           ...  ...     ...     ...        ...
1424          NaN            NaN           NaN  ...     NaN     NaN        NaN
1425          NaN            NaN           NaN  ...     NaN     NaN        NaN
1426          NaN            NaN           NaN  ...     NaN     NaN        NaN
1427          NaN            NaN           NaN  ...     NaN     NaN        NaN
1428          NaN            NaN           NaN  ...     NaN     NaN        NaN

[71945 rows x 32 columns]
Therefore i have dropped columns with 75% missing values using dropna thresh-

As a rule of thumb, when the data goes missing on 60–70 percent of the variable, dropping the variable should be considered and Filled the missing data with the mean and median value if it's a numerical variable and for categorical values Filled the missing data with mode

[16]
0s
#View data frame after dropping attributes with 70% NaN values- we remain with32 attributes for our Analysis
df

[17]
0s
#Check missing values (second time dealing with a big data)
df.isnull().sum()
Task_ID                  18943
Order_ID                 18943
Relationship             18943
Team_Name                18943
Task_Type                18943
Agent_ID                 18943
Distance(m)              18943
Total_Time_Taken(min)    18943
Pick_up_From             18943
Start_Before             18943
Complete_Before          18943
Completion_Time          18943
Task_Status              18943
Ref_Images               18943
Rating                   18943
Review                   18968
Latitude                 18943
Longitude                18943
Promo_Applied            19052
Custom_Template_ID       18943
Task_Details_QTY         18943
Task_Details_AMOUNT      18943
Special_Instructions     18943
Tip                       5272
Delivery_Charges         18943
Discount                  5272
Subtotal                 18943
Payment_Type             18943
Task_Category            18943
Earning                  18943
Pricing                  18943
Agent_Name               26962
dtype: int64
[19]
0s
#Percentage of column missing values
df.isna().mean() * 100
Task_ID                  26.329835
Order_ID                 26.329835
Relationship             26.329835
Team_Name                26.329835
Task_Type                26.329835
Agent_ID                 26.329835
Distance(m)              26.329835
Total_Time_Taken(min)    26.329835
Pick_up_From             26.329835
Start_Before             26.329835
Complete_Before          26.329835
Completion_Time          26.329835
Task_Status              26.329835
Ref_Images               26.329835
Rating                   26.329835
Review                   26.364584
Latitude                 26.329835
Longitude                26.329835
Promo_Applied            26.481340
Custom_Template_ID       26.329835
Task_Details_QTY         26.329835
Task_Details_AMOUNT      26.329835
Special_Instructions     26.329835
Tip                       7.327820
Delivery_Charges         26.329835
Discount                  7.327820
Subtotal                 26.329835
Payment_Type             26.329835
Task_Category            26.329835
Earning                  26.329835
Pricing                  26.329835
Agent_Name               37.475850
dtype: float64
From the name of the attributes, we can distinguish that three columns are spectific for identifying each order. We need to drop them from the dataset as they are not going to be useful for the analysis.

[20]
0s
#df.drop('Task_ID',
 # axis=1, inplace=True)
#df.drop('Custom_Template_ID',
 # axis=1, inplace=True)
df.drop('Agent_ID',
  axis=1, inplace=True)
/usr/local/lib/python3.7/dist-packages/pandas/core/frame.py:4913: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  errors=errors,
[22]
0s
#3. We replace the NaN values with the **Mode** and **median** for categorical and Nu,erical attributes respectively
# Update missing values
df['Tip'].fillna((df['Tip'].mode()), inplace=True)
df['Discount'].fillna((df['Discount'].mode()), inplace=True)
df['Order_ID'].fillna((df['Order_ID'].mode()), inplace=True)
df['Relationship'].fillna((df['Relationship'].mode()), inplace=True)
df['Team_Name'].fillna((df['Team_Name'].mode()), inplace=True)
df['Task_Type'].fillna((df['Task_Type'].mode()), inpla

[23]
0s
len(df['Rating'].unique())
7
[24]
0s
df.Rating.value_counts()
0    44784
0     7999
-      109
5       90
5       20
1        4
Name: Rating, dtype: int64
[25]
0s
(df['Rating'].unique())
array([0, 5, '0', nan, '5', '-', '1'], dtype=object)
[27]
1s
df = df.dropna()#drop all NaN values
df['Rating'] = df['Rating'].astype(int)
[28]
0s
#Check missing values (confirm if we have any missing values)
df.isnull().sum()
Task_ID                  0
Order_ID                 0
Relationship             0
Team_Name                0
Task_Type                0
Distance(m)              0
Total_Time_Taken(min)    0
Pick_up_From             0
Start_Before             0
Complete_Before          0
Completion_Time          0
Task_Status              0
Ref_Images               0
Rating                   0
Review                   0
Latitude                 0
Longitude                0
Promo_Applied            0
Custom_Template_ID       0
Task_Details_QTY         0
Task_Details_AMOUNT      0
Special_Instructions     0
Tip                      0
Delivery_Charges         0
Discount                 0
Subtotal                 0
Payment_Type             0
Task_Category            0
Earning                  0
Pricing                  0
Agent_Name               0
dtype: int64
[29]
0s
len(df['Payment_Type'].unique())
32
[30]
0s
df.Payment_Type.value_counts()
-                                                      24297
CASH                                                   20309
Pay Later                                                190
paybill                                                   10
Paybill                                                    7
PAYBILL                                                    7
mpesa                                                      7
To pay later                                               6
To pay later.                                              2
To pay after 3days                                         1
To pay in the evening.                                     1
CASH 4000 Bal 1489                                         1
To pay later on paybill                                    1
CASH 3629. Paybill 14516                                   1
To pay later in the evening                                1
paid 4000 on paybill, 16005 in Cash                        1
CASH 3149 Paybill 800                                      1
Not paid                                                   1
To pay on paybill                                          1
To pay on paybill later                                    1
Paybill 2000 bal 1900 to be paid tomorrow                  1
paybill later                                              1
paid ksh2000 to paybill,bal to clear in the evening        1
CASH298 Paybill 2100                                       1
CASH1500 paybill 1463                                      1
CASH 700 Paybill 2700                                      1
CASH 2610 paybill 1000                                     1
paid 3000on paybill                                        1
CASH 2410 PAYBILL 2700                                     1
CASHaa                                                     1
paid less 80 bob                                           1
CASH 400 mpesa 1400                                        1
Name: Payment_Type, dtype: int64
[31]
0s
# View columns available for analysis
df.columns
Index(['Task_ID', 'Order_ID', 'Relationship', 'Team_Name', 'Task_Type',
       'Distance(m)', 'Total_Time_Taken(min)', 'Pick_up_From', 'Start_Before',
       'Complete_Before', 'Completion_Time', 'Task_Status', 'Ref_Images',
       'Rating', 'Review', 'Latitude', 'Longitude', 'Promo_Applied',
       'Custom_Template_ID', 'Task_Details_QTY', 'Task_Details_AMOUNT',
       'Special_Instructions', 'Tip', 'Delivery_Charges', 'Discount',
       'Subtotal', 'Payment_Type', 'Task_Category', 'Earning', 'Pricing',
       'Agent_Name'],
      dtype='object')
[32]
0s
# converting Subtotal column  format from string to numeric, and filling its Null values with 0.

df['Subtotal'] = pd.to_numeric(df['Subtotal'], downcast="float", errors='coerce')

df = df.fillna(0)
df['Subtotal'] 
0            0.0
0            0.0
1         4700.0
2        19500.0
3         4350.0
          ...   
44978     6240.0
44979     3990.0
44980        0.0
44981        0.0
44982     7040.0
Name: Subtotal, Length: 44858, dtype: float32
[33]
0s
len(df['Earning'].unique())
2
[34]
0s
df.Earning.value_counts()
-    28754
0    16104
Name: Earning, dtype: int64
[35]
0s
df['Earning'] = pd.to_numeric(df['Earning'], downcast="float", errors='coerce')

df = df.fillna(1)
df['Earning']
0        1.0
0        1.0
1        1.0
2        1.0
3        1.0
        ... 
44978    1.0
44979    1.0
44980    1.0
44981    1.0
44982    1.0
Name: Earning, Length: 44858, dtype: float32
[56]
1s
len(df['Task_Details_QTY'].unique())
74
[57]
0s
df.Task_Details_QTY.value_counts()
1      25857
2       7273
3       2811
5       2284
10      1639
       ...  
63         1
31         1
49         1
180        1
128        1
Name: Task_Details_QTY, Length: 74, dtype: int64
Exploratory Data Analysis

Because our target variable is retention, i should analyse the data focusing on how the different features are related to this variable.I set customers who have purchased 2 or more quantities as retained while customers who had purchased less than 2 quantities as churned/exited customers

[193]
1s
df.Task_Details_QTY
0         5.0
0         1.0
1        12.0
2        10.0
3         1.0
         ... 
44978     1.0
44979    10.0
44980     1.0
44981     1.0
44982    10.0
Name: Task_Details_QTY, Length: 44858, dtype: float32
[ ]
df.Task_Details_QTY
[397]
0s
df['Task_Details_QTY']= pd.to_numeric(df['Task_Details_QTY'], downcast="float", errors='coerce')
df['Task_Details_QTY']
0        100.0
0        100.0
1        100.0
2        100.0
3        100.0
         ...  
44978    100.0
44979    100.0
44980    100.0
44981    100.0
44982    100.0
Name: Task_Details_QTY, Length: 44858, dtype: float32
[396]
0s
retained = df[df['Task_Details_QTY'] >=2 ]['Task_Details_QTY'].count() / df.shape[0] * 100
exited = df[df['Task_Details_QTY'] <2 ]['Task_Details_QTY'].count() / df.shape[0] * 100
retained
100.0
[392]
0s
exited
0.0
[140]
0s
df['Task_Details_QTY']
0         5.0
0         1.0
1        12.0
2        10.0
3         1.0
         ... 
44978     1.0
44979    10.0
44980     1.0
44981     1.0
44982    10.0
Name: Task_Details_QTY, Length: 44858, dtype: float32
[179]
1s
sns.countplot(df['Task_Details_QTY'],label="Count")

[99]
0s
df[df['Task_Details_QTY'] >=2 ]['Task_Details_QTY'].count()
19000
[39]
0s
df.Task_Status.value_counts()

Completed     40050
Cancelled      2897
Failed         1838
Unassigned       71
Assigned          1
Declined          1
Name: Task_Status, dtype: int64
[40]
0s
len(df['Task_Status'].unique())
6
TASK 3:USE ML MODEL(S) TO PREDICT CUSTOMER RETENTION

Double-click (or enter) to edit

Train/Test split

[196]
0s
X = df.loc[:, df.columns != 'Subtotal']
Y = df.Subtotal
Y
0            0.0
0            0.0
1         4700.0
2        19500.0
3         4350.0
          ...   
44978     6240.0
44979     3990.0
44980        0.0
44981        0.0
44982     7040.0
Name: Subtotal, Length: 44858, dtype: float32
[ ]

[113]
0s
# PCA
cols = ['Rating','Subtotal','Task_Status','Task_Details_QTY']
# standardize data
model = pipeline.Pipeline([('std', preprocessing.StandardScaler()),
                            ('pca', decomposition.PCA(random_state=42))])
X = pd.get_dummies(df[cols], drop_first=True).fillna(0)
X_pca = model.fit_transform(X)
pca = model.named_steps['pca']
[328]
1s
X_pca.shape
(44858, 8)
[115]
0s
model.steps
[('std', StandardScaler()), ('pca', PCA(random_state=42))]
[116]
0s
plt.plot(np.cumsum(pca.explained_variance_ratio_))

[117]
0s
np.cumsum(pca.explained_variance_ratio_)
array([0.24382554, 0.4427343 , 0.57435877, 0.69964437, 0.82464793,
       0.94955318, 0.99998586, 1.        ])
[118]
0s
# Find columns that most influence components
comps = pd.DataFrame(pca.components_, columns=X.columns)
pca_cols = set()
num_comps = 2
for i in range(num_comps):
    parts = comps.iloc[i][comps.iloc[i].abs() > .2]
    pca_cols.update(set(parts.index))
pca_cols
{'Subtotal',
 'Task_Details_QTY',
 'Task_Status_Cancelled',
 'Task_Status_Completed',
 'Task_Status_Failed'}
[119]
0s
# How original columns impact each component
pd.DataFrame(pca.components_, columns=X.columns).loc[:,list(pca_cols)].T

[120]
0s
# add coloring
(pd.DataFrame(pca.components_, columns=X.columns)
 .loc[:,list(pca_cols)]
 .T
 .style.background_gradient(cmap='RdBu', axis=0)
)

[121]
0s
#Alternate view
fig, ax = plt.subplots(figsize=(8,8))
plt.imshow(pd.DataFrame(pca.components_, columns=X.columns).loc[:5, list(pca_cols)].T, 
           cmap='PiYG', vmin=-.4, vmax=.4)
plt.yticks(range(len(pca_cols)), list(pca_cols))
plt.colorbar()

[178]
0s
# Check Task status for churning and non churning customers
pd.DataFrame(df.groupby('Task_Details_QTY')['Task_Status'].describe())

[377]
0s
x = [df.Task_Details_QTY,df.Task_Details_QTY]
y = [df.Subtotal,df.Subtotal]

[378]
0s
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
[379]
0s
x = np.array(x)
y = np.array(y)
Support Vector Machine-We are ready to build different models looking for the best fit. Predicting customer churn is a binary classification problem: Customers are either lost or retained in a given period of time.

[380]
0s
param_grid_svm = {
    'C': [0.5, 100, 150],
    'kernel': ['rbf'],
    'gamma': [0.1, 0.01, 0.001],
    'probability': [True]
}
[372]
0s
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
[373]
svm_first = SVC()
svm_grid = GridSearchCV(svm_first, param_grid=param_grid_svm, cv=1, verbose=1, n_jobs=-2)

[381]
0s
x
array([[ 5.,  1., 12., ...,  1.,  1., 10.],
       [ 5.,  1., 12., ...,  1.,  1., 10.]], dtype=float32)
[382]
0s
y
array([[   0.,    0., 4700., ...,    0.,    0., 7040.],
       [   0.,    0., 4700., ...,    0.,    0., 7040.]], dtype=float32)
[398]
0s
#svm_grid.fit(x, y)

check
0s
completed at 12:27 AM
Made 1 formatting edit on line 5
Could not connect to the reCAPTCHA service. Please check your internet connection and reload to get a reCAPTCHA challenge.
Automatic saving failed. This file was updated remotely or in another tab.
