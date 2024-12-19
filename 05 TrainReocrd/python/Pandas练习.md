## Pandas
- pandas可以很好的处理除了数值型的其他数据
- Pandas库给予Python Numpy库开发而来，因此，它可以与Python的科学计算库配合使用。Pandas提供了两种数据结构
    -   Series（一维数组结构）
    -   DataFrame（二维数组结构）

### Series
- series是一种类似与一维数组的对象，由两个部分组成：
    - values: 一组数据（ndarray类型）
    - index：相关的数据索引标签


```python
import numpy as np
import pandas as pd
from pandas import Series as ss
```

#### Series的创建
    - 由列表或numpy数组创建
    - 由字典创建


```python
#使用数组充当数据源创建的Series容器
s1 = Series(data=[1,2,3,4,5])
s1
```




    0    1
    1    2
    2    3
    3    4
    4    5
    dtype: int64




```python
# 使用一维的numpy数组充当数据源创建Series容器
s2 = Series(data=np.random.randint(0,100,size=(5,)))
print(s2)
s2=s2.astype('int8')
print(s2)
```

    0    54
    1    25
    2    70
    3     5
    4    22
    dtype: int32
    0    54
    1    25
    2    70
    3     5
    4    22
    dtype: int8
    


```python
#字典充当数据源，字典的key会作为Series的索引，字典的value作为Series的元素
dic={
    'name':'张三',
    'age':30,
    'address':'BJ'
}

s3=Series(data=dic)
s3
```




    name       张三
    age        30
    address    BJ
    dtype: object



#### 使用列表创建Series


```python
import pandas as pd
import numpy as np

ser1=pd.Series([np.random.randint(0,100,size=(5,))])
ser1
```




    0    [36, 80, 67, 53, 55]
    dtype: object



#### Series的索引
    - 隐形索引：默认形式的索引（0,1,2.....）；
    - 显式索引：自定义的索引，可以通过index参数设置显示索引；


```python
s4=Series(data=[1,2,3],index=['A','B','C'])
s4
```




    A    1
    B    2
    C    3
    dtype: int64




```python
s4=s4.astype('int8')
s4
```




    A    1
    B    2
    C    3
    dtype: int8



##### 显示索引的作用：添加了数据的可读性
 - Series的索引和切片


```python
s5 =Series(data=[1,2,3,4,5],index=['a','b','c','d','e'])
s5
```




    a    1
    b    2
    c    3
    d    4
    e    5
    dtype: int64




```python
s5[0],s5['a'],s5.a
```

    C:\Users\lsyon\AppData\Local\Temp\ipykernel_2144\2799027360.py:1: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      s5[0],s5['a'],s5.a
    




    (1, 1, 1)




```python
s5[0:3]
```




    a    1
    b    2
    c    3
    dtype: int64



#### Series的常用属性
    - shape
    - size
    - index
    -values


```python
s5.index
```




    Index(['a', 'b', 'c', 'd', 'e'], dtype='object')




```python
s5.values
```




    array([1, 2, 3, 4, 5], dtype=int64)




```python
s5.size
```




    5




```python
s5.shape
```




    (5,)



#### Series 的常用方法
- head()     查看前n个元素
- tail()
- unique(),nunique(),values_counts()
- isnull(),notnull()
- add(),sub(),mul(),div()


```python
s=Series(data=[1,2,None,3,None,4,5,None,6])
s
```




    0    1.0
    1    2.0
    2    NaN
    3    3.0
    4    NaN
    5    4.0
    6    5.0
    7    NaN
    8    6.0
    dtype: float64




```python
s.head(n=3)  #查看前n个元素
```




    0    1.0
    1    2.0
    2    NaN
    dtype: float64




```python
s.tail(n=2)  #查看后n个元素
```




    7    NaN
    8    6.0
    dtype: float64




```python
s.unique() #去除重复的元素
```




    array([ 1.,  2., nan,  3.,  4.,  5.,  6.])




```python
s.nunique() #统计去重后的元素个数
```




    6




```python
s.value_counts() #统计每个元素出现的次数，注意：不会统计空值出现的次数
```




    1.0    1
    2.0    1
    3.0    1
    4.0    1
    5.0    1
    6.0    1
    Name: count, dtype: int64




```python
s.isnull()   #检测每个元素是否为空，为空则返回True，否则返回False

```




    0    False
    1    False
    2     True
    3    False
    4     True
    5    False
    6    False
    7     True
    8    False
    dtype: bool




```python
s.notnull()
```




    0     True
    1     True
    2    False
    3     True
    4    False
    5     True
    6     True
    7    False
    8     True
    dtype: bool




```python
#清洗s中存在的空值(可以将布尔值作为s的索引来使用，只会保留True对应的元素，忽略False对应的元素)
s[[True,True,False,True,False,True,True,False,True]]

```




    0    1.0
    1    2.0
    3    3.0
    5    4.0
    6    5.0
    8    6.0
    dtype: float64




```python
s[s.notnull()]
```




    0    1.0
    1    2.0
    3    3.0
    5    4.0
    6    5.0
    8    6.0
    dtype: float64



### DataFrame
- DataFrame是一个【表格型】的数据结构；
- DataFrame由按一定顺序排列的多列数据组成；
- DataFrame即有行索引，也有列索引；
    - 行索引：index
    - 列索引：columns
    - 值：values

#### DataFrame的创建
    - ndarray创建
    - 字典创建


```python
from pandas import DataFrame 
df1 =DataFrame(data=np.random.randint(0,100,size=(5,6)))
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75</td>
      <td>36</td>
      <td>32</td>
      <td>54</td>
      <td>17</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>6</td>
      <td>90</td>
      <td>1</td>
      <td>64</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>43</td>
      <td>59</td>
      <td>2</td>
      <td>22</td>
      <td>74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92</td>
      <td>63</td>
      <td>31</td>
      <td>66</td>
      <td>3</td>
      <td>58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>58</td>
      <td>88</td>
      <td>45</td>
      <td>31</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
dic = {
    'name':['zhangsan','lisi','wangwu'],
    'age':[30,40,50],
    'salary':[1000,2000,3000]
}
df2=DataFrame(data=dic,index=['a','b','c'])
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>zhangsan</td>
      <td>30</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>b</th>
      <td>lisi</td>
      <td>40</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>c</th>
      <td>wangwu</td>
      <td>50</td>
      <td>3000</td>
    </tr>
  </tbody>
</table>
</div>



#### DataFrame的属性
- values、columns、index、shape


```python
df2.values
```




    array([['zhangsan', 30, 1000],
           ['lisi', 40, 2000],
           ['wangwu', 50, 3000]], dtype=object)




```python
df2.index
```




    Index(['a', 'b', 'c'], dtype='object')




```python
df2.columns

```




    Index(['name', 'age', 'salary'], dtype='object')




```python
df2.shape
```




    (3, 3)



#### DataFrame索引操作
- 对行进行索引
- 对列进行索引
- 对元素进行索引


```python
dic = {
    'name':['zhangsan','lisi','wangwu'],
    'salary':[1000,2000,3000],
    'age':[20,30,33],
    'dep':['sale','opt','sale']
}

df = DataFrame(data=dic,index=['A','B','C'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>salary</th>
      <th>age</th>
      <th>dep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>zhangsan</td>
      <td>1000</td>
      <td>20</td>
      <td>sale</td>
    </tr>
    <tr>
      <th>B</th>
      <td>lisi</td>
      <td>2000</td>
      <td>30</td>
      <td>opt</td>
    </tr>
    <tr>
      <th>C</th>
      <td>wangwu</td>
      <td>3000</td>
      <td>33</td>
      <td>sale</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['name'] # 索引取列
```




    A    zhangsan
    B        lisi
    C      wangwu
    Name: name, dtype: object




```python
df[['name','age']]  # 索引取多列
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>zhangsan</td>
      <td>20</td>
    </tr>
    <tr>
      <th>B</th>
      <td>lisi</td>
      <td>30</td>
    </tr>
    <tr>
      <th>C</th>
      <td>wangwu</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>




```python
#索引取行
df.loc['A']  # 显示索引
```




    name      zhangsan
    salary        1000
    age             20
    dep           sale
    Name: A, dtype: object




```python
df.iloc[0] # 隐式索引
```




    name      zhangsan
    salary        1000
    age             20
    dep           sale
    Name: A, dtype: object




```python
# 取元素
df.iloc[0,1]
```




    1000



#### DataFarme的切片操作
- 对行进行切片
- 对列进行切片


```python
#切行
df[0:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>salary</th>
      <th>age</th>
      <th>dep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>zhangsan</td>
      <td>1000</td>
      <td>20</td>
      <td>sale</td>
    </tr>
    <tr>
      <th>B</th>
      <td>lisi</td>
      <td>2000</td>
      <td>30</td>
      <td>opt</td>
    </tr>
  </tbody>
</table>
</div>




```python
#切列
df.iloc[:,0:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>zhangsan</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>lisi</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>C</th>
      <td>wangwu</td>
      <td>3000</td>
    </tr>
  </tbody>
</table>
</div>



#### 时间类型数据的转换
- pd.to_datetime(col)


```python
dic = {
    'name':['zhangsan','lisi','wangwu'],
    'hire_date':["2022-01-10",'2021-11-11','2022-09-09'],
    'salary':[1000,2000,3000]
}

df =DataFrame(data=dic)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>hire_date</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>zhangsan</td>
      <td>2022-01-10</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lisi</td>
      <td>2021-11-11</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>wangwu</td>
      <td>2022-09-09</td>
      <td>3000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看每一列的数据类型
df.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   name       3 non-null      object
     1   hire_date  3 non-null      object
     2   salary     3 non-null      int64 
    dtypes: int64(1), object(2)
    memory usage: 204.0+ bytes
    


```python
df['hire_date']= pd.to_datetime(df['hire_date'])
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype         
    ---  ------     --------------  -----         
     0   name       3 non-null      object        
     1   hire_date  3 non-null      datetime64[ns]
     2   salary     3 non-null      int64         
    dtypes: datetime64[ns](1), int64(1), object(1)
    memory usage: 204.0+ bytes
    

#### 时间类型的dt属性操作


```python
df['hire_date'].dt.year
```




    0    2022
    1    2021
    2    2022
    Name: hire_date, dtype: int32




```python
df['hire_date'].dt.month
```




    0     1
    1    11
    2     9
    Name: hire_date, dtype: int32




```python
df['hire_date'].dt.day
```




    0    10
    1    11
    2     9
    Name: hire_date, dtype: int32




```python
#df['hire_date'].dt.week
df['hire_date'].dt.isocalendar().week
```




    0     2
    1    45
    2    36
    Name: week, dtype: UInt32



#### 将某一列设置为行索引
   - df.set_index()


```python
df.set_index('hire_date')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>salary</th>
    </tr>
    <tr>
      <th>hire_date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-10</th>
      <td>zhangsan</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>2021-11-11</th>
      <td>lisi</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>2022-09-09</th>
      <td>wangwu</td>
      <td>3000</td>
    </tr>
  </tbody>
</table>
</div>



 #### 在 DataFrame 中使用“isin”过滤多行



```python
import pandas as pd
 
employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp003</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>Statistician</td>
      <td>2018-02-26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees.loc[employees['Occupation'].isin(['Chemist','Programmer'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>City</th>
      <th>Age</th>
      <th>EmpCode</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>London</td>
      <td>23</td>
      <td>Emp001</td>
      <td>John</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>Toronto</td>
      <td>40</td>
      <td>Emp005</td>
      <td>Mark</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees.loc[employees['Age']==23]
employees.loc[employees['Age']<30]
employees.loc[(employees['Occupation'] != 'Statistician')&(employees['Name']=='John')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees.loc[(employees['Occupation'] == 'Chemist') | (employees['Name'] =='John') & (employees['Age'] < 30)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



#### 迭代 DataFrame 的行和列



```python
for index,col in employees.iterrows():
    print(col['Name'],'---',col['Age'])
```

    John --- 23
    Doe --- 24
    William --- 34
    Spark --- 29
    Mark --- 40
    


```python
for row in employees.itertuples(index=True,name='Pandas'):
    print(getattr(row,'Name'),'--',getattr(row,'Age'))
```

    John -- 23
    Doe -- 24
    William -- 34
    Spark -- 29
    Mark -- 40
    


```python
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp003</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>Statistician</td>
      <td>2018-02-26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



#### 如何通过名称或索引删除 DataFrame 的列


```python
employees.drop('Age',axis=1,inplace=True)
```


```python
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp003</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>Statistician</td>
      <td>2018-02-26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>Programmer</td>
      <td>2018-03-16</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees.drop(employees.columns[[0,1]],axis=1,inplace=True)
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Occupation</th>
      <th>Date Of Join</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chemist</td>
      <td>2018-01-25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Statistician</td>
      <td>2018-01-26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Statistician</td>
      <td>2018-01-26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Statistician</td>
      <td>2018-02-26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Programmer</td>
      <td>2018-03-16</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

#### 向 DataFrame 中新增列



```python
employees[ 'Age']=[23, 24, 34, 29, 40]
employees['EmpCode']=['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005']
employees['Name']= ['John', 'Doe', 'William', 'Spark', 'Mark']
employees['City'] = ['London', 'Tokyo', 'Sydney', 'London', 'Toronto']
```

#### 如何从 DataFrame 中获取列标题列表



```python
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>City</th>
      <th>Age</th>
      <th>EmpCode</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>London</td>
      <td>23</td>
      <td>Emp001</td>
      <td>John</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>Tokyo</td>
      <td>24</td>
      <td>Emp002</td>
      <td>Doe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>Sydney</td>
      <td>34</td>
      <td>Emp003</td>
      <td>William</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Statistician</td>
      <td>2018-02-26</td>
      <td>London</td>
      <td>29</td>
      <td>Emp004</td>
      <td>Spark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>Toronto</td>
      <td>40</td>
      <td>Emp005</td>
      <td>Mark</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(list(employees))
print(list(employees.columns.values))
print(list(employees.columns.tolist()))
```

    ['Occupation', 'Date Of Join', 'City', 'Age', 'EmpCode', 'Name']
    ['Occupation', 'Date Of Join', 'City', 'Age', 'EmpCode', 'Name']
    ['Occupation', 'Date Of Join', 'City', 'Age', 'EmpCode', 'Name']
    

#### 如何随机生成 DataFrame


```python
import pandas as pd
import numpy as np
 
np.random.seed(5)
 
df_random = pd.DataFrame(np.random.randint(100, size=(10, 6)),
                         columns=list('ABCDEF'),
                         index=['Row-{}'.format(i) for i in range(10)])
df_random

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Row-0</th>
      <td>99</td>
      <td>78</td>
      <td>61</td>
      <td>16</td>
      <td>73</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Row-1</th>
      <td>62</td>
      <td>27</td>
      <td>30</td>
      <td>80</td>
      <td>7</td>
      <td>76</td>
    </tr>
    <tr>
      <th>Row-2</th>
      <td>15</td>
      <td>53</td>
      <td>80</td>
      <td>27</td>
      <td>44</td>
      <td>77</td>
    </tr>
    <tr>
      <th>Row-3</th>
      <td>75</td>
      <td>65</td>
      <td>47</td>
      <td>30</td>
      <td>84</td>
      <td>86</td>
    </tr>
    <tr>
      <th>Row-4</th>
      <td>18</td>
      <td>9</td>
      <td>41</td>
      <td>62</td>
      <td>1</td>
      <td>82</td>
    </tr>
    <tr>
      <th>Row-5</th>
      <td>16</td>
      <td>78</td>
      <td>5</td>
      <td>58</td>
      <td>0</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Row-6</th>
      <td>4</td>
      <td>36</td>
      <td>51</td>
      <td>27</td>
      <td>31</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Row-7</th>
      <td>68</td>
      <td>38</td>
      <td>83</td>
      <td>19</td>
      <td>18</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Row-8</th>
      <td>30</td>
      <td>62</td>
      <td>11</td>
      <td>67</td>
      <td>65</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Row-9</th>
      <td>3</td>
      <td>91</td>
      <td>78</td>
      <td>27</td>
      <td>29</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>



#### 如何选择 DataFrame 的多个列


```python
df=employees[['Name','EmpCode','Age']]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>EmpCode</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Emp001</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Doe</td>
      <td>Emp002</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>William</td>
      <td>Emp003</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spark</td>
      <td>Emp004</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mark</td>
      <td>Emp005</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



#### 如何将字典转换为DataFrame


```python
import pandas as pd
 
data = ({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   })
print(data)
pd.DataFrame(data)
```

    {'Age': [30, 20, 22, 40, 32, 28, 39], 'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black', 'Red'], 'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese', 'Melon', 'Beans'], 'Height': [165, 70, 120, 80, 180, 172, 150], 'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2], 'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']}
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>Green</td>
      <td>Lamb</td>
      <td>70</td>
      <td>8.3</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>5</th>
      <td>28</td>
      <td>Black</td>
      <td>Melon</td>
      <td>172</td>
      <td>9.5</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>6</th>
      <td>39</td>
      <td>Red</td>
      <td>Beans</td>
      <td>150</td>
      <td>2.2</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>



#### 使用ioc进行切片


```python


import pandas as pd
 
df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>20</td>
      <td>Green</td>
      <td>Lamb</td>
      <td>70</td>
      <td>8.3</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>28</td>
      <td>Black</td>
      <td>Melon</td>
      <td>172</td>
      <td>9.5</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>39</td>
      <td>Red</td>
      <td>Beans</td>
      <td>150</td>
      <td>2.2</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['Penelope']
```




    Age          40
    Color     White
    Food      Apple
    Height       80
    Score       3.3
    State        AL
    Name: Penelope, dtype: object




```python
df.loc[['Cornelia','Jane','Dean']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cornelia</th>
      <td>39</td>
      <td>Red</td>
      <td>Beans</td>
      <td>150</td>
      <td>2.2</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['Aaron':'Dean']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
  </tbody>
</table>
</div>



#### 检查DataFrame中是否是空的


```python
import pandas as pd
 
df = pd.DataFrame()
 
if df.empty:
    print('DataFrame is empty!')
```

    DataFrame is empty!
    

#### 在创建DataFrame时指定索引和列名称


```python
import pandas as pd
 
values = ["India", "Canada", "Australia",
          "Japan", "Germany", "France"]
 
code = ["IND", "CAN", "AUS", "JAP", "GER", "FRA"]

df=pd.DataFrame(values,index=code,columns=['Country'])
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IND</th>
      <td>India</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>Canada</td>
    </tr>
    <tr>
      <th>AUS</th>
      <td>Australia</td>
    </tr>
    <tr>
      <th>JAP</th>
      <td>Japan</td>
    </tr>
    <tr>
      <th>GER</th>
      <td>Germany</td>
    </tr>
    <tr>
      <th>FRA</th>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>



#### 使用iloc进行切片


```python

import pandas as pd

df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>20</td>
      <td>Green</td>
      <td>Lamb</td>
      <td>70</td>
      <td>8.3</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>28</td>
      <td>Black</td>
      <td>Melon</td>
      <td>172</td>
      <td>9.5</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>39</td>
      <td>Red</td>
      <td>Beans</td>
      <td>150</td>
      <td>2.2</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[4]
```




    Age           32
    Color       Gray
    Food      Cheese
    Height       180
    Score        1.8
    State         AK
    Name: Dean, dtype: object




```python
df.iloc[[2,-2]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>28</td>
      <td>Black</td>
      <td>Melon</td>
      <td>172</td>
      <td>9.5</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:5:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>



#### iloc 和 loc 的区别
* loc索引器还可以进行布尔选择，例如，如果我们想查找Age小于30的所有行并仅返回Color和Height列，我们可以执行以下操作。 我们可以用iloc复制它，但我们不能将它传递给一个布尔系列，必须将布尔系列转换为numpy数组
* loc从索引中获取具有特定便签的行（或列）
* iloc在索引中的特定位置获取行（或列）（因此它只需要整数）


```python
import pandas as pd
 
df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>20</td>
      <td>Green</td>
      <td>Lamb</td>
      <td>70</td>
      <td>8.3</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>28</td>
      <td>Black</td>
      <td>Melon</td>
      <td>172</td>
      <td>9.5</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>39</td>
      <td>Red</td>
      <td>Beans</td>
      <td>150</td>
      <td>2.2</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df['Age']<30,['Color','Height']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Color</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nick</th>
      <td>Green</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>Red</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>Black</td>
      <td>172</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[(df['Age']<30).values,[1,3]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Color</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nick</th>
      <td>Green</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>Red</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>Black</td>
      <td>172</td>
    </tr>
  </tbody>
</table>
</div>



#### 使用时间索引创建空 DataFrame


```python
import datetime
import pandas as pd
 
todays_date = datetime.datetime.now().date()
index = pd.date_range(todays_date, periods=10, freq='D')
 
columns = ['A', 'B', 'C']

df=pd.DataFrame(index=index,columns=columns)
df=df.fillna(0)
df
```

    C:\Users\lsyon\AppData\Local\Temp\ipykernel_19764\2472431921.py:10: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df=df.fillna(0)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-12-26</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### 如何改变DataFrame列的排序


```python


import pandas as pd

df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>20</td>
      <td>Green</td>
      <td>Lamb</td>
      <td>70</td>
      <td>8.3</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>28</td>
      <td>Black</td>
      <td>Melon</td>
      <td>172</td>
      <td>9.5</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>39</td>
      <td>Red</td>
      <td>Beans</td>
      <td>150</td>
      <td>2.2</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_order = [3, 2, 1, 4, 5, 0]
df=df[df.columns[new_order]]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>Food</th>
      <th>Color</th>
      <th>Score</th>
      <th>State</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>165</td>
      <td>Steak</td>
      <td>Blue</td>
      <td>4.6</td>
      <td>NY</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>70</td>
      <td>Lamb</td>
      <td>Green</td>
      <td>8.3</td>
      <td>TX</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>120</td>
      <td>Mango</td>
      <td>Red</td>
      <td>9.0</td>
      <td>FL</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>80</td>
      <td>Apple</td>
      <td>White</td>
      <td>3.3</td>
      <td>AL</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>180</td>
      <td>Cheese</td>
      <td>Gray</td>
      <td>1.8</td>
      <td>AK</td>
      <td>32</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>172</td>
      <td>Melon</td>
      <td>Black</td>
      <td>9.5</td>
      <td>TX</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>150</td>
      <td>Beans</td>
      <td>Red</td>
      <td>2.2</td>
      <td>TX</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
df=df.reindex(['State', 'Color', 'Age', 'Food', 'Score', 'Height'],axis=1)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Color</th>
      <th>Age</th>
      <th>Food</th>
      <th>Score</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>NY</td>
      <td>Blue</td>
      <td>30</td>
      <td>Steak</td>
      <td>4.6</td>
      <td>165</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>TX</td>
      <td>Green</td>
      <td>20</td>
      <td>Lamb</td>
      <td>8.3</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>FL</td>
      <td>Red</td>
      <td>22</td>
      <td>Mango</td>
      <td>9.0</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>AL</td>
      <td>White</td>
      <td>40</td>
      <td>Apple</td>
      <td>3.3</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>AK</td>
      <td>Gray</td>
      <td>32</td>
      <td>Cheese</td>
      <td>1.8</td>
      <td>180</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>TX</td>
      <td>Black</td>
      <td>28</td>
      <td>Melon</td>
      <td>9.5</td>
      <td>172</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>TX</td>
      <td>Red</td>
      <td>39</td>
      <td>Beans</td>
      <td>2.2</td>
      <td>150</td>
    </tr>
  </tbody>
</table>
</div>



#### 检查DataFrame 列的数据类型


```python

import pandas as pd
 
df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Color</th>
      <th>Food</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>Blue</td>
      <td>Steak</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>20</td>
      <td>Green</td>
      <td>Lamb</td>
      <td>70</td>
      <td>8.3</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>Red</td>
      <td>Mango</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>White</td>
      <td>Apple</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>Gray</td>
      <td>Cheese</td>
      <td>180</td>
      <td>1.8</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>28</td>
      <td>Black</td>
      <td>Melon</td>
      <td>172</td>
      <td>9.5</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>39</td>
      <td>Red</td>
      <td>Beans</td>
      <td>150</td>
      <td>2.2</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    Age         int64
    Color      object
    Food       object
    Height      int64
    Score     float64
    State      object
    dtype: object



#### 更换DataFrame指定列的数据类型


```python
df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                   'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                             'Red'],
                   'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                            'Melon', 'Beans'],
                   'Height': [165, 70, 120, 80, 180, 172, 150],
                   'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df.dtypes
```




    Age         int64
    Color      object
    Food       object
    Height      int64
    Score     float64
    State      object
    dtype: object




```python
df['Age']=df['Age'].astype(str)
df.dtypes
```




    Age        object
    Color      object
    Food       object
    Height      int64
    Score     float64
    State      object
    dtype: object



#### 将列数据类型转换为DateTime类型


```python
import pandas as pd
 
df = pd.DataFrame({'DateOFBirth': [1349720105, 1349806505, 1349892905,
                                   1349979305, 1350065705, 1349792905,
                                   1349730105],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOFBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1349720105</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>1349806505</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1349892905</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>1349979305</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>1350065705</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1349792905</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1349730105</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df)
df['DateOFBirth']=pd.to_datetime(df['DateOFBirth'],unit='s')
print(df)
```

               DateOFBirth State
    Jane        1349720105    NY
    Nick        1349806505    TX
    Aaron       1349892905    FL
    Penelope    1349979305    AL
    Dean        1350065705    AK
    Christina   1349792905    TX
    Cornelia    1349730105    TX
                      DateOFBirth State
    Jane      2012-10-08 18:15:05    NY
    Nick      2012-10-09 18:15:05    TX
    Aaron     2012-10-10 18:15:05    FL
    Penelope  2012-10-11 18:15:05    AL
    Dean      2012-10-12 18:15:05    AK
    Christina 2012-10-09 14:28:25    TX
    Cornelia  2012-10-08 21:01:45    TX
    

#### 将DataFrame 列从floats转为int


```python
import pandas as pd
 
df = pd.DataFrame({'DailyExp': [75.7, 56.69, 55.69, 96.5, 84.9, 110.5,
                                58.9],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DailyExp</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>75.70</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>56.69</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>55.69</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>96.50</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>84.90</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>110.50</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>58.90</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.dtypes)
print(df)
df['DailyExp']=df['DailyExp'].astype(int)
print(df.dtypes)
print(df)
```

    DailyExp    float64
    State        object
    dtype: object
               DailyExp State
    Jane          75.70    NY
    Nick          56.69    TX
    Aaron         55.69    FL
    Penelope      96.50    AL
    Dean          84.90    AK
    Christina    110.50    TX
    Cornelia      58.90    TX
    DailyExp     int32
    State       object
    dtype: object
               DailyExp State
    Jane             75    NY
    Nick             56    TX
    Aaron            55    FL
    Penelope         96    AL
    Dean             84    AK
    Christina       110    TX
    Cornelia         58    TX
    

#### 将date列转换为DateTime类型


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],                   
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
```


```python
print(df.dtypes)
df['DateOfBirth']=df['DateOfBirth'].astype('datetime64[ns]')
print(df.dtypes)
```

    DateOfBirth    object
    State          object
    dtype: object
    DateOfBirth    datetime64[ns]
    State                  object
    dtype: object
    

#### 两个DataFarme合并


```python
import pandas as pd
 
df1 = pd.DataFrame({'Age': [30, 20, 22, 40], 'Height': [165, 70, 120, 80],
                    'Score': [4.6, 8.3, 9.0, 3.3], 'State': ['NY', 'TX',
                                                             'FL', 'AL']},
                   index=['Jane', 'Nick', 'Aaron', 'Penelope'])
 
df2 = pd.DataFrame({'Age': [32, 28, 39], 'Color': ['Gray', 'Black', 'Red'],
                    'Food': ['Cheese', 'Melon', 'Beans'],
                    'Score': [1.8, 9.5, 2.2], 'State': ['AK', 'TX', 'TX']},
                   index=['Dean', 'Christina', 'Cornelia'])
df3 = pd.concat((df1,df2))
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
      <th>Color</th>
      <th>Food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>165.0</td>
      <td>4.6</td>
      <td>NY</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>20</td>
      <td>70.0</td>
      <td>8.3</td>
      <td>TX</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>120.0</td>
      <td>9.0</td>
      <td>FL</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>80.0</td>
      <td>3.3</td>
      <td>AL</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>32</td>
      <td>NaN</td>
      <td>1.8</td>
      <td>AK</td>
      <td>Gray</td>
      <td>Cheese</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>28</td>
      <td>NaN</td>
      <td>9.5</td>
      <td>TX</td>
      <td>Black</td>
      <td>Melon</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>39</td>
      <td>NaN</td>
      <td>2.2</td>
      <td>TX</td>
      <td>Red</td>
      <td>Beans</td>
    </tr>
  </tbody>
</table>
</div>



#### DateFrame 末尾添加额外的行


```python
import pandas as pd
 
employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})
employees

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp003</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>Statistician</td>
      <td>2018-02-26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees.loc[len(employees)]=['Emp006', 'Sunny', 'Programmer', '2018-01-25',45]
print(employees)
```

      EmpCode     Name    Occupation Date Of Join  Age
    0  Emp001     John       Chemist   2018-01-25   23
    1  Emp002      Doe  Statistician   2018-01-26   24
    2  Emp003  William  Statistician   2018-01-26   34
    3  Emp004    Spark  Statistician   2018-02-26   29
    4  Emp005     Mark    Programmer   2018-03-16   40
    5  Emp006    Sunny    Programmer   2018-01-25   45
    6  Emp006    Sunny    Programmer   2018-01-25   45
    

#### 为指定索引添加新行


```python
import pandas as pd
 
employees = pd.DataFrame(
    data={'Name': ['John Doe', 'William Spark'],
          'Occupation': ['Chemist', 'Statistician'],
          'Date Of Join': ['2018-01-25', '2018-01-26'],
          'Age': [23, 24]},
    index=['Emp001', 'Emp002'],
    columns=['Name', 'Occupation', 'Date Of Join', 'Age'])
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Emp001</th>
      <td>John Doe</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Emp002</th>
      <td>William Spark</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees.loc['Emp003']=['Sunny', 'Programmer', '2018-01-25', 45]
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Emp001</th>
      <td>John Doe</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Emp002</th>
      <td>William Spark</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Emp003</th>
      <td>Sunny</td>
      <td>Programmer</td>
      <td>2018-01-25</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>



#### 如何使用for循环添加行


```python
import pandas as pd

cols=['Zip']
lst=[]
zip=32100

for a in range(10):
    lst.append([zip])
    zip=zip+10

df=pd.DataFrame(lst,columns=cols)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Zip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32110</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32140</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32150</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32160</td>
    </tr>
    <tr>
      <th>7</th>
      <td>32170</td>
    </tr>
    <tr>
      <th>8</th>
      <td>32180</td>
    </tr>
    <tr>
      <th>9</th>
      <td>32190</td>
    </tr>
  </tbody>
</table>
</div>



#### 在DataFrame顶部添加一行


```python
import pandas as pd
 
employees = pd.DataFrame({
    'EmpCode': ['Emp002', 'Emp003', 'Emp004'],
    'Name': ['John', 'Doe', 'William'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26'],
    'Age': [23, 24, 34]})

employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp002</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp003</td>
      <td>Doe</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp004</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>




```python
line = pd.DataFrame({'Name': 'Dean', 'Age': 45, 'EmpCode': 'Emp001','Date Of Join': '2018-02-26', 'Occupation': 'Chemist'}, index=[0])
line
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
      <th>EmpCode</th>
      <th>Date Of Join</th>
      <th>Occupation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dean</td>
      <td>45</td>
      <td>Emp001</td>
      <td>2018-02-26</td>
      <td>Chemist</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees=pd.concat([line,employees.iloc[:]]).reset_index(drop=True)
employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
      <th>EmpCode</th>
      <th>Date Of Join</th>
      <th>Occupation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dean</td>
      <td>45</td>
      <td>Emp001</td>
      <td>2018-02-26</td>
      <td>Chemist</td>
    </tr>
    <tr>
      <th>1</th>
      <td>John</td>
      <td>23</td>
      <td>Emp002</td>
      <td>2018-01-25</td>
      <td>Chemist</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Doe</td>
      <td>24</td>
      <td>Emp003</td>
      <td>2018-01-26</td>
      <td>Statistician</td>
    </tr>
    <tr>
      <th>3</th>
      <td>William</td>
      <td>34</td>
      <td>Emp004</td>
      <td>2018-01-26</td>
      <td>Statistician</td>
    </tr>
  </tbody>
</table>
</div>



#### 如何向DataFrame中动态添加行


```python
import pandas as pd
 
df = pd.DataFrame(columns=['Name', 'Age'])
df.loc[1,'Name']='Rocky'
df.loc[1,'Age']=23
print(df,'\n')


df.loc[2,'Name']='Sunny'
print(df)
```

        Name Age
    1  Rocky  23 
    
        Name  Age
    1  Rocky   23
    2  Sunny  NaN
    

#### 在任意位置插入行


```python
import pandas as pd 
df=pd.DataFrame(columns=['Name','Age'])
df.loc[1,'Name']='Rocky'
df.loc[1,'Age']=21
df.loc[2,'Name']='Sunny'
df.loc[2,'Age']=22
df.loc[3, 'Name'] = 'Mark'
df.loc[3, 'Age'] = 25
df.loc[4, 'Name'] = 'Taylor'
df.loc[4, 'Age'] = 28
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Rocky</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sunny</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mark</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Taylor</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
line = pd.DataFrame({"Name": "Jack", "Age": 24}, index=[2.5])

df = pd.concat([line,df],ignore_index=False)

df=df.sort_index().reset_index(drop=True)

df=df.reindex(['Name','Age'],axis=1)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rocky</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sunny</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jack</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jack</td>
      <td>24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jack</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mark</td>
      <td>25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Taylor</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



#### 使用时间戳索引向DataFrame中添加行


```python
import pandas as pd
 
df = pd.DataFrame(columns=['Name', 'Age'])
df.loc['2014-05-01 18:47:05', 'Name'] = 'Rocky'
df.loc['2014-05-01 18:47:05', 'Age'] = 21
df.loc['2014-05-02 18:47:05', 'Name'] = 'Sunny'
df.loc['2014-05-02 18:47:05', 'Age'] = 22
df.loc['2014-05-03 18:47:05', 'Name'] = 'Mark'
df.loc['2014-05-03 18:47:05', 'Age'] = 25
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-05-01 18:47:05</th>
      <td>Rocky</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2014-05-02 18:47:05</th>
      <td>Sunny</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2014-05-03 18:47:05</th>
      <td>Mark</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
line = pd.to_datetime('2014-05-01 18:50:05',format='%Y-%m-%d %H:%M:%S')
new_order=pd.DataFrame([['Bunny',26]],columns=['Name','Age'],index=[line])
df=pd.concat([df,pd.DataFrame(new_order)],ignore_index=False)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-05-01 18:47:05</th>
      <td>Rocky</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2014-05-02 18:47:05</th>
      <td>Sunny</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2014-05-03 18:47:05</th>
      <td>Mark</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2014-05-01 18:50:05</th>
      <td>Bunny</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



#### 为不同的行填充缺失值


```python
import pandas as pd
 
a = {'A': 10, 'B': 20}
b = {'B': 30, 'C': 40, 'D': 50}

df1=pd.DataFrame(a,index=[0])
df2=pd.DataFrame(b,index=[1])
df=pd.DataFrame()
df=pd.concat([df1,df2]).fillna(0)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>20</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>30</td>
      <td>40.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
</div>



####  concat和combine_first示例


```python
import pandas as pd
 
a = {'A': 10, 'B': 20}
b = {'B': 30, 'C': 40, 'D': 50}
 
df1 = pd.DataFrame(a, index=[0])
df2 = pd.DataFrame(b, index=[1])

d2 = pd.concat([df1, df2]).fillna(0)
print(d2,'\n')
d3=pd.DataFrame()
d3=d3.combine_first(df1).combine_first(df2).fillna(0)
print(d3,'\n')
```

          A   B     C     D
    0  10.0  20   0.0   0.0
    1   0.0  30  40.0  50.0 
    
          A   B     C     D
    0  10.0  20   0.0   0.0
    1   0.0  30  40.0  50.0 
    
    

#### 获取行和列的平均值


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5, 5, 0, 0]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Mean Basket']=df.mean(axis=1)  # 求行
df.loc['Mean Fruit']=df.mean()  #求列
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
      <th>Mean Basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10.000000</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>40.000000</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7.000000</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>28.000000</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5.000000</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>Mean Fruit</th>
      <td>7.333333</td>
      <td>13.0</td>
      <td>17.0</td>
      <td>22.666667</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Apple'].mean()
```




    7.333333333333333



#### 获取行和列的总和


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5, 5, 0, 0]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Sum Basket']=df.sum(axis=1)  # 求行
df.loc['Sum Fruit']=df.sum()  #求列
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
      <th>Sum Basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Sum Fruit</th>
      <td>22</td>
      <td>39</td>
      <td>51</td>
      <td>68</td>
      <td>180</td>
    </tr>
  </tbody>
</table>
</div>



#### 连接两列


```python
df = pd.DataFrame(columns=['Name', 'Age'])
 
df.loc[1, 'Name'] = 'Rocky'
df.loc[1, 'Age'] = 21
 
df.loc[2, 'Name'] = 'Sunny'
df.loc[2, 'Age'] = 22
 
df.loc[3, 'Name'] = 'Mark'
df.loc[3, 'Age'] = 25
 
df.loc[4, 'Name'] = 'Taylor'
df.loc[4, 'Age'] = 28
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Rocky</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sunny</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mark</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Taylor</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Employee']=df['Name'].map(str)+'-'+df['Age'].map(str)
df=df.reindex(['Employee'],axis=1)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Employee</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Rocky-21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sunny-22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mark-25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Taylor-28</td>
    </tr>
  </tbody>
</table>
</div>



#### 过滤包含某字符串的行


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1976-01-01</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>1986-06-01</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1=df[df['State'].str.contains("TX")]  # contains 用于检查每个元素中是否包含指定的字符串
#df[df['State']=='TX']
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>



#### 过滤索引中包含某字符串的行


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Pane', 'Aaron', 'Penelope', 'Frane',
                         'Christina', 'Cornelia'])
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1976-01-01</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>1986-06-01</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Frane</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index=df.index.astype(str)
df1=df[df.index.str.contains('ane')]  # contains 用于检查每个元素中是否包含指定的字符串
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Frane</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
  </tbody>
</table>
</div>



#### 使用AND 运算符过滤包含特定字符串值的行


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Pane', 'Aaron', 'Penelope', 'Frane',
                         'Christina', 'Cornelia'])
df
```


```python
df.index=df.index.astype(str)
df1=df[df.index.str.contains('ane') & df['State'].str.contains('TX')]
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>



#### 查找包含某个字符串的所有行


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Pane', 'Aaron', 'Penelope', 'Frane',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1976-01-01</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>1986-06-01</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Frane</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index=df.index.astype(str)
df1=df[df.index.str.contains('ane') | df['State'].str.contains('TX')]
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Frane</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>



#### *如果行中的值包含字符串，则创建于字符串相等的另一列


```python
import pandas as pd
import numpy as np
 
df = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Accountant', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>Accountant</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp003</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>Statistician</td>
      <td>2018-02-26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Department'] = np.where(
    df.Occupation.str.contains('Chemist'),"Science",
    np.where(df.Occupation.str.contains('Statistician'),"Economics",
             np.where(df.Occupation.str.contains('Programmer'),"Computer","General")))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
      <th>Department</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
      <td>Science</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>Accountant</td>
      <td>2018-01-26</td>
      <td>24</td>
      <td>General</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp003</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>34</td>
      <td>Economics</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>Statistician</td>
      <td>2018-02-26</td>
      <td>29</td>
      <td>Economics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>40</td>
      <td>Computer</td>
    </tr>
  </tbody>
</table>
</div>



#### 计算pandas group 中每组的行数


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5, 5, 0, 0],
                   [6, 6, 6, 6], [8, 8, 8, 8], [5, 5, 0, 0]],
                  columns=['Apple', 'Orange', 'Rice', 'Oil'],
                  index=['Basket1', 'Basket2', 'Basket3',
                         'Basket4', 'Basket5', 'Basket6'])
 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Rice</th>
      <th>Oil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Apple','Orange','Rice','Oil']].groupby(['Apple']).agg(['mean','count'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Orange</th>
      <th colspan="2" halign="left">Rice</th>
      <th colspan="2" halign="left">Oil</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>count</th>
      <th>mean</th>
      <th>count</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Apple</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14.0</td>
      <td>1</td>
      <td>21.0</td>
      <td>1</td>
      <td>28.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>1</td>
      <td>8.0</td>
      <td>1</td>
      <td>8.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>1</td>
      <td>40.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### 检查字符串是否在DataFrme中


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Pane', 'Aaron', 'Penelope', 'Frane',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1976-01-01</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>1986-06-01</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Frane</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
#any()是 Python 的内置函数及pandas中Series等数据结构的方法，用于判断可迭代对象或数据结构中的元素是否至少有一个为True，
#与检查所有元素是否为True的all()方法相对，且有一定性能优势。
if df['State'].str.contains('TX').any():
    print("TX is there")
```

    TX is there
    

#### 从DataFrame列中获取唯一行值


```python
import pandas as pd
 
df = pd.DataFrame({'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['State'].unique()
```




    array(['NY', 'TX', 'FL', 'AL', 'AK'], dtype=object)



#### 计算DataFrame列的不同值的个数


```python
import pandas as pd
 
df = pd.DataFrame({'Age': [30, 20, 22, 40, 20, 30, 20, 25],
                    'Height': [165, 70, 120, 80, 162, 72, 124, 81],
                    'Score': [4.6, 8.3, 9.0, 3.3, 4, 8, 9, 3],
                    'State': ['NY', 'TX', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                   index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Jaane', 'Nicky', 'Armour', 'Ponting'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>20</td>
      <td>70</td>
      <td>8.3</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>20</td>
      <td>162</td>
      <td>4.0</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
      <td>8.0</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>124</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
      <td>3.0</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.Age.value_counts()
```




    Age
    20    3
    30    2
    22    1
    40    1
    25    1
    Name: count, dtype: int64



#### *删除具有重复索引的行


```python
import pandas as pd

df = pd.DataFrame({'Age': [30, 30, 22, 40, 20, 30, 20, 25],
                   'Height': [165, 165, 120, 80, 162, 72, 124, 81],
                   'Score': [4.6, 4.6, 9.0, 3.3, 4, 8, 9, 3],
                   'State': ['NY', 'NY', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>20</td>
      <td>162</td>
      <td>4.0</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
      <td>8.0</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>124</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
      <td>3.0</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.reset_index().drop_duplicates(subset='index',keep='first').set_index('index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>165</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>22</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>80</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>20</td>
      <td>162</td>
      <td>4.0</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
      <td>8.0</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>124</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
      <td>3.0</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>



#### 删除某些列具有重复值的行


```python
df = pd.DataFrame({'Age': [30, 40, 30, 40, 30, 30, 20, 25],
                   'Height': [120, 162, 120, 120, 120, 72, 120, 81],
                   'Score': [4.6, 4.6, 9.0, 3.3, 4, 8, 9, 3],
                   'State': ['NY', 'NY', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>120</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>40</td>
      <td>162</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>30</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>120</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>30</td>
      <td>120</td>
      <td>4.0</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
      <td>8.0</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
      <td>3.0</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.reset_index().drop_duplicates(subset=['Age','Height'],keep='first').set_index('index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>120</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>40</td>
      <td>162</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>120</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
      <td>8.0</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
      <td>3.0</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>



#### 从DataFrame单元格中获取值


```python
import pandas as pd

df = pd.DataFrame({'Age': [30, 40, 30, 40, 30, 30, 20, 25],
                   'Height': [120, 162, 120, 120, 120, 72, 120, 81],
                   'Score': [4.6, 4.6, 9.0, 3.3, 4, 8, 9, 3],
                   'State': ['NY', 'NY', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>120</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>40</td>
      <td>162</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>30</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>120</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>30</td>
      <td>120</td>
      <td>4.0</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
      <td>8.0</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
      <td>3.0</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['Nicky','Age']
```




    30



#### 使用DataFrame中的条件索引获取单元格上的标量值


```python
import pandas as pd

df = pd.DataFrame({'Age': [30, 40, 30, 40, 30, 30, 20, 25],
                   'Height': [120, 162, 120, 120, 120, 72, 120, 81],
                   'Score': [4.6, 4.6, 9.0, 3.3, 4, 8, 9, 3],
                   'State': ['NY', 'NY', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Score</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>120</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>40</td>
      <td>162</td>
      <td>4.6</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>30</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>120</td>
      <td>3.3</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>30</td>
      <td>120</td>
      <td>4.0</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
      <td>8.0</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>120</td>
      <td>9.0</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
      <td>3.0</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df['Age']==20,'Height'].values[0]
```




    120




```python
df.loc[df['Age']==30,'State'].values[2]
```




    'NY'



#### 设置DataFrame的特定单元格值


```python
import pandas as pd
 
df = pd.DataFrame({'Age': [30, 40, 30, 40, 30, 30, 20, 25],
                   'Height': [120, 162, 120, 120, 120, 72, 120, 81]},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>40</td>
      <td>162</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>30</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>30</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iat[0,0]=90
df.iat[0,1]=91
df.iat[1,1]=92
df.iat[2,1]=93
df.iat[7,1]=99
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>90</td>
      <td>91</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>40</td>
      <td>92</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>30</td>
      <td>93</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>30</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
</div>



#### 从DataFrame行获取单元格值


```python
import pandas as pd
 
df = pd.DataFrame({'Age': [30, 40, 30, 40, 30, 30, 20, 25],
                   'Height': [120, 162, 120, 120, 120, 72, 120, 81]},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>30</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>40</td>
      <td>162</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>30</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>40</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Jaane</th>
      <td>30</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Nicky</th>
      <td>30</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Armour</th>
      <td>20</td>
      <td>120</td>
    </tr>
    <tr>
      <th>Ponting</th>
      <td>25</td>
      <td>81</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df.Age==30,'Height'].tolist()
```




    [120, 120, 120, 72]




```python
df.loc[df['Age']==30,'Height']
```




    Jane     120
    Aaron    120
    Jaane    120
    Nicky     72
    Name: Height, dtype: int64



#### 用字典替换DataFrame列中的值


```python
import pandas as pd
 
df = pd.DataFrame({'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
dict = {"NY": 1, "TX": 2, "FL": 3, "AL": 4, "AK": 5}
df1=df.replace({'State':dict})
df1
```

    C:\Users\lsyon\AppData\Local\Temp\ipykernel_19764\2386565715.py:2: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df1=df.replace({'State':dict})
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



#### 统一基于某一列的一列数值


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],                   
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Nick</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1976-01-01</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>1986-06-01</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Dean</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('State').DateOfBirth.nunique()
```




    State
    AK    1
    AL    1
    FL    1
    NY    1
    TX    3
    Name: DateOfBirth, dtype: int64



#### 处理DataFrame中的缺失值


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5,]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.notnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



#### 删除包含任何缺失数据的行


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5,]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 删除DataFrame中缺失数据的列


```python
import pandas as pd

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5,]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



#### 按照降序对索引值进行排序


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Pane', 'Aaron', 'Penelope', 'Frane',
                         'Christina', 'Cornelia'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1976-01-01</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>Penelope</th>
      <td>1986-06-01</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Frane</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index(ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateOfBirth</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Penelope</th>
      <td>1986-06-01</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>Pane</th>
      <td>1999-05-12</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Jane</th>
      <td>1986-11-11</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>Frane</th>
      <td>1983-06-04</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>Cornelia</th>
      <td>1999-07-09</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Christina</th>
      <td>1990-03-07</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>Aaron</th>
      <td>1976-01-01</td>
      <td>FL</td>
    </tr>
  </tbody>
</table>
</div>



#### 按降序对列进行排序


```python
import pandas as pd
 
employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})
employees
```


```python
employees.sort_index(axis=1,ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Occupation</th>
      <th>Name</th>
      <th>EmpCode</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chemist</td>
      <td>John</td>
      <td>Emp001</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Statistician</td>
      <td>Doe</td>
      <td>Emp002</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Statistician</td>
      <td>William</td>
      <td>Emp003</td>
      <td>2018-01-26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Statistician</td>
      <td>Spark</td>
      <td>Emp004</td>
      <td>2018-02-26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Programmer</td>
      <td>Mark</td>
      <td>Emp005</td>
      <td>2018-03-16</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



#### 使用rank方法查找DataFrame中元素的排名


```python
import pandas as pd

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5, 5, 0, 0]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rank()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 在多列上设置索引


```python
import pandas as pd
 
employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})

employees
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Occupation</th>
      <th>Date Of Join</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emp001</td>
      <td>John</td>
      <td>Chemist</td>
      <td>2018-01-25</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emp003</td>
      <td>William</td>
      <td>Statistician</td>
      <td>2018-01-26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>Statistician</td>
      <td>2018-02-26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>Programmer</td>
      <td>2018-03-16</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
employees.set_index(['Occupation','Age'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>EmpCode</th>
      <th>Name</th>
      <th>Date Of Join</th>
    </tr>
    <tr>
      <th>Occupation</th>
      <th>Age</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chemist</th>
      <th>23</th>
      <td>Emp001</td>
      <td>John</td>
      <td>2018-01-25</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Statistician</th>
      <th>24</th>
      <td>Emp002</td>
      <td>Doe</td>
      <td>2018-01-26</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Emp003</td>
      <td>William</td>
      <td>2018-01-26</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Emp004</td>
      <td>Spark</td>
      <td>2018-02-26</td>
    </tr>
    <tr>
      <th>Programmer</th>
      <th>40</th>
      <td>Emp005</td>
      <td>Mark</td>
      <td>2018-03-16</td>
    </tr>
  </tbody>
</table>
</div>



#### 确定DataFrame的周期索引和列


```python
import pandas as pd
 
values = ["India", "Canada", "Australia",
          "Japan", "Germany", "France"]
pidx=pd.period_range('2015-01-01',periods=6)
df=pd.DataFrame(values,index=pidx,columns=['Country'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01</th>
      <td>India</td>
    </tr>
    <tr>
      <th>2015-01-02</th>
      <td>Canada</td>
    </tr>
    <tr>
      <th>2015-01-03</th>
      <td>Australia</td>
    </tr>
    <tr>
      <th>2015-01-04</th>
      <td>Japan</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>Germany</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>



#### 导入CSV指定特定索引


```python
import pandas as pd
 
df = pd.read_csv('test.csv', index_col="DateTime")
df
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[323], line 3
          1 import pandas as pd
    ----> 3 df = pd.read_csv('test.csv', index_col="DateTime")
          4 df
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1026, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
       1013 kwds_defaults = _refine_defaults_read(
       1014     dialect,
       1015     delimiter,
       (...)
       1022     dtype_backend=dtype_backend,
       1023 )
       1024 kwds.update(kwds_defaults)
    -> 1026 return _read(filepath_or_buffer, kwds)
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:626, in _read(filepath_or_buffer, kwds)
        623     return parser
        625 with parser:
    --> 626     return parser.read(nrows)
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1923, in TextFileReader.read(self, nrows)
       1916 nrows = validate_integer("nrows", nrows)
       1917 try:
       1918     # error: "ParserBase" has no attribute "read"
       1919     (
       1920         index,
       1921         columns,
       1922         col_dict,
    -> 1923     ) = self._engine.read(  # type: ignore[attr-defined]
       1924         nrows
       1925     )
       1926 except Exception:
       1927     self.close()
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\c_parser_wrapper.py:333, in CParserWrapper.read(self, nrows)
        330     data = {k: v for k, (i, v) in zip(names, data_tups)}
        332     names, date_data = self._do_date_conversions(names, data)
    --> 333     index, column_names = self._make_index(date_data, alldata, names)
        335 return index, column_names, date_data
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\base_parser.py:371, in ParserBase._make_index(self, data, alldata, columns, indexnamerow)
        368     index = None
        370 elif not self._has_complex_date_col:
    --> 371     simple_index = self._get_simple_index(alldata, columns)
        372     index = self._agg_index(simple_index)
        373 elif self._has_complex_date_col:
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\base_parser.py:403, in ParserBase._get_simple_index(self, data, columns)
        401 index = []
        402 for idx in self.index_col:
    --> 403     i = ix(idx)
        404     to_remove.append(i)
        405     index.append(data[i])
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\base_parser.py:398, in ParserBase._get_simple_index.<locals>.ix(col)
        396 if not isinstance(col, str):
        397     return col
    --> 398 raise ValueError(f"Index {col} invalid")
    

    ValueError: Index DateTime invalid


#### 将 DataFrame 写入 csv


```python
import pandas as pd
 
df = pd.DataFrame({'DateOfBirth': ['1986-11-11', '1999-05-12', '1976-01-01',
                                   '1986-06-01', '1983-06-04', '1990-03-07',
                                   '1999-07-09'],
                   'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                   },
                  index=['Jane', 'Pane', 'Aaron', 'Penelope', 'Frane',
                         'Christina', 'Cornelia'])
 
df.to_csv('test.csv', encoding='utf-8', index=True)


```

#### 使用 Pandas 读取 csv 文件的特定列


```python
import pandas as pd
 
df = pd.read_csv("test.csv", usecols = ['DateOfBirth','State'])
print(df)
```

      DateOfBirth State
    0  1986-11-11    NY
    1  1999-05-12    TX
    2  1976-01-01    FL
    3  1986-06-01    AL
    4  1983-06-04    AK
    5  1990-03-07    TX
    6  1999-07-09    TX
    

#### Pandas 获取 CSV 列的列表


```python
import pandas as pd

cols=list(pd.read_csv('test.csv',nrows=1))
cols
```




    ['Unnamed: 0', 'DateOfBirth', 'State']



#### 找到列的最大值


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df['Apple'].idxmax()]
```




    Apple     55
    Orange    15
    Banana     8
    Pear      12
    Name: Basket3, dtype: int64



#### 使用查询方法进行复杂条件选择


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df.query('Apple >  50 & Orange<= 15 & Banana < 15 & Pear ==  12').index]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



#### 检查Pandas中是否存在列


```python
import pandas as pd

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
if 'Apple' in df.columns:
    print('Yes')
else:
    print('No')

```

    Yes
    


```python

if set(['Apple','Orange']).issubset(df.columns):
    print('Yes')
else:
    print('No')
```

    Yes
    

#### 为特定列从DataFrame中查找n-smallest和n-largest值


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.nsmallest(2,['Apple'])  #nsmallest:用于获取属性指定数量的最小值对应的行
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.nlargest(3,['Apple'])#nlargest: 用于获取属性指定数量的最大值对应的行
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



#### 从DataFram中查找所有列的最小值和最大值


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Apple','Orange','Banana','Pear']].min()
```




    Apple     5
    Orange    1
    Banana    1
    Pear      2
    dtype: int64




```python
df[['Apple','Orange','Banana','Pear']].max()
```




    Apple     55
    Orange    20
    Banana    30
    Pear      40
    dtype: int64



#### 在Dataframe中找到最小值和最大值所在的索引位置


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Apple','Orange','Banana','Pear']].idxmin()
```




    Apple     Basket6
    Orange    Basket5
    Banana    Basket4
    Pear      Basket6
    dtype: object




```python
df[['Apple','Orange','Banana','Pear']].idxmax()
```




    Apple     Basket3
    Orange    Basket1
    Banana    Basket1
    Pear      Basket1
    dtype: object



#### 计算DataFrame Columms 的累计乘积和累积总和


```python
import pandas as pd

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Apple','Orange','Banana','Pear']].cumprod()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>70</td>
      <td>280</td>
      <td>630</td>
      <td>1120</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>3850</td>
      <td>4200</td>
      <td>5040</td>
      <td>13440</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>57750</td>
      <td>58800</td>
      <td>5040</td>
      <td>107520</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>404250</td>
      <td>58800</td>
      <td>5040</td>
      <td>860160</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>2021250</td>
      <td>235200</td>
      <td>45360</td>
      <td>1720320</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Apple','Orange','Banana','Pear']].cumsum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>17</td>
      <td>34</td>
      <td>51</td>
      <td>68</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>72</td>
      <td>49</td>
      <td>59</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>87</td>
      <td>63</td>
      <td>60</td>
      <td>88</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>94</td>
      <td>64</td>
      <td>61</td>
      <td>96</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>99</td>
      <td>68</td>
      <td>70</td>
      <td>98</td>
    </tr>
  </tbody>
</table>
</div>



#### 汇总统计


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.500000</td>
      <td>11.333333</td>
      <td>11.666667</td>
      <td>16.333333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.180719</td>
      <td>7.257180</td>
      <td>11.587349</td>
      <td>14.555640</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.000000</td>
      <td>6.500000</td>
      <td>2.750000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.500000</td>
      <td>14.000000</td>
      <td>8.500000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.750000</td>
      <td>14.750000</td>
      <td>18.000000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55.000000</td>
      <td>20.000000</td>
      <td>30.000000</td>
      <td>40.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Apple']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.180719</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### 查找DataFrame的均值、中值和众数


```python
import pandas as pd

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean()
```




    Apple     16.500000
    Orange    11.333333
    Banana    11.666667
    Pear      16.333333
    dtype: float64




```python
df.median()
```




    Apple      8.5
    Orange    14.0
    Banana     8.5
    Pear      10.0
    dtype: float64




```python
df.mode()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



#### 测量DataFrame列的方差和标准差


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#方差
df.var()
```




    Apple     367.900000
    Orange     52.666667
    Banana    134.266667
    Pear      211.866667
    dtype: float64




```python
df.std()#标准差
```




    Apple     19.180719
    Orange     7.257180
    Banana    11.587349
    Pear      14.555640
    dtype: float64



#### *计算DataFrame列之间的协方差


```python
import pandas as pd 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.cov()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Apple</th>
      <td>367.9</td>
      <td>47.600000</td>
      <td>-40.200000</td>
      <td>-35.000000</td>
    </tr>
    <tr>
      <th>Orange</th>
      <td>47.6</td>
      <td>52.666667</td>
      <td>54.333333</td>
      <td>77.866667</td>
    </tr>
    <tr>
      <th>Banana</th>
      <td>-40.2</td>
      <td>54.333333</td>
      <td>134.266667</td>
      <td>154.933333</td>
    </tr>
    <tr>
      <th>Pear</th>
      <td>-35.0</td>
      <td>77.866667</td>
      <td>154.933333</td>
      <td>211.866667</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.Apple.cov(df.Orange)
```




    47.6



#### *计算Pandas中两个DataFram对象直接的相关性


```python
import pandas as pd

df1 = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.corr()  #计算了其自身各列之间的相关性
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Apple</th>
      <td>1.000000</td>
      <td>0.341959</td>
      <td>-0.180874</td>
      <td>-0.125364</td>
    </tr>
    <tr>
      <th>Orange</th>
      <td>0.341959</td>
      <td>1.000000</td>
      <td>0.646122</td>
      <td>0.737144</td>
    </tr>
    <tr>
      <th>Banana</th>
      <td>-0.180874</td>
      <td>0.646122</td>
      <td>1.000000</td>
      <td>0.918606</td>
    </tr>
    <tr>
      <th>Pear</th>
      <td>-0.125364</td>
      <td>0.737144</td>
      <td>0.918606</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame([[52, 54, 58, 41], [14, 24, 51, 78], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 17, 18, 98], [15, 34, 29, 52]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>52</td>
      <td>54</td>
      <td>58</td>
      <td>41</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>14</td>
      <td>24</td>
      <td>51</td>
      <td>78</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>17</td>
      <td>18</td>
      <td>98</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>15</td>
      <td>34</td>
      <td>29</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.corrwith(other=df1) #corrwith方法则用于计算两个数据帧之间的相关性，可以指定一个数据帧作为参数，计算另一个数据帧与它的相关性
```




    Apple     0.678775
    Orange    0.354993
    Banana    0.920872
    Pear      0.076919
    dtype: float64



#### 计算DataFrame 列的每个单元格的百分比变化


```python
import pandas as pd
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10</td>
      <td>20</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>7</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>55</td>
      <td>15</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15</td>
      <td>14</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pct_change()方法通过计算当前元素与前一个元素的百分比变化来实现。具体计算公式为：(当前值 - 前一个值) / 前一个值 * 100
df[['Apple']].pct_change()[:3] 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>-0.300000</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>6.857143</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pct_change()[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>-0.300000</td>
      <td>-0.300000</td>
      <td>-0.300000</td>
      <td>-0.300000</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>6.857143</td>
      <td>0.071429</td>
      <td>-0.619048</td>
      <td>-0.571429</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>-0.727273</td>
      <td>-0.066667</td>
      <td>-0.875000</td>
      <td>-0.333333</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>-0.533333</td>
      <td>-0.928571</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### 在Pandas中向前和向后填充DataFrame列的缺失值


```python
import pandas as pd
 
df = pd.DataFrame([[10, 30, 40], [], [15, 8, 12],
                   [15, 14, 1, 8], [7, 8], [5, 4, 1]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.ffill() #向前填充缺失值
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>10.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.bfill() #向后填充缺失值
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### *在Pandas中使用非分层索引，使用Stacking


```python
import pandas as pd
 
df = pd.DataFrame([[10, 30, 40], [], [15, 8, 12],
                   [15, 14, 1, 8], [7, 8], [5, 4, 1]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.stack(level=0)
```




    Basket1  Apple     10.0
             Orange    30.0
             Banana    40.0
    Basket3  Apple     15.0
             Orange     8.0
             Banana    12.0
    Basket4  Apple     15.0
             Orange    14.0
             Banana     1.0
             Pear       8.0
    Basket5  Apple      7.0
             Orange     8.0
    Basket6  Apple      5.0
             Orange     4.0
             Banana     1.0
    dtype: float64



#### 使用分层索引对Pandas进行拆分


```python
import pandas as pd

df = pd.DataFrame([[10, 30, 40], [], [15, 8, 12],
                   [15, 14, 1, 8], [7, 8], [5, 4, 1]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Orange</th>
      <th>Banana</th>
      <th>Pear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basket1</th>
      <td>10.0</td>
      <td>30.0</td>
      <td>40.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket3</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket4</th>
      <td>15.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Basket5</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Basket6</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.unstack(level=-1)
```




    Apple   Basket1    10.0
            Basket2     NaN
            Basket3    15.0
            Basket4    15.0
            Basket5     7.0
            Basket6     5.0
    Orange  Basket1    30.0
            Basket2     NaN
            Basket3     8.0
            Basket4    14.0
            Basket5     8.0
            Basket6     4.0
    Banana  Basket1    40.0
            Basket2     NaN
            Basket3    12.0
            Basket4     1.0
            Basket5     NaN
            Basket6     1.0
    Pear    Basket1     NaN
            Basket2     NaN
            Basket3     NaN
            Basket4     8.0
            Basket5     NaN
            Basket6     NaN
    dtype: float64



#### Pandas 获取 HTML 页面上 table 数据


```python
import pandas as pd
df= pd.read_html("https://www.runoob.com/python3/python3-conditional-statements.html")
df
```




    [  操作符            描述
     0   <            小于
     1  <=         小于或等于
     2   >            大于
     3  >=         大于或等于
     4  ==  等于，比较两个值是否相等
     5  !=           不等于,
         类型              False          True
     0   布尔        False(与0等价)    True(与1等价)
     1   数值             0, 0.0         非零的数值
     2  字符串       '', ""(空字符串)         非空字符串
     3   容器  [], (), {}, set()  至少有一个元素的容器对象
     4  NaN                NaN       非None对象]




```python

```
