# pandas基础

## 1、pandas数据结构介绍

* pandas采用了很多numpy的代码风格，但最大的不同在于pandas是用来处理表格型或异质型数据的，而numpy更适合处理同质型的数值类数组数据
* pandas有两个常用的数据结构：Series和DataFrame
* Series是一种一维的数组型对象，它包含了一个值序列和数据标签(索引)，默认生成的索引是从0到N-1(N是数据长度)，可以通过values属性和index属性分别获得Series对象的值和索引，从另一个角度考虑Series，可以认为它是一个长度固定且有序的字典，因为它将索引值和数据值按位置配对
* pandas中使用isnull和notnull函数来检查缺失数据
* 在数学操作中series会自动对齐索引
* series的索引可以通过按位置赋值的方式改变
* DataFrame表示的是矩阵的数据表，DataFrame既有行索引也有列索引，它可以被视为一个共享相同索引的Series的字典
* 尽管DataFrame是二维的，但可以利用分层索引在DataFrame中展现更高维度的数据
* 通过df[column]索引列，df.loc df.iloc索引行 def df[column]可以删除列 df[column]选取的列是数据的视图，而不是拷贝，如需复制，则应当显式地使用copy方法
* 如果嵌套字典被赋值给DataFrame，pandas会将字典的键作为列，将内部字典的键作为行索引，包含Series的字典也可以用于构造DataFrame，原理类似
* 可以利用frame.T实现对DataFrame的转置操作
* df.values会将数据以二维ndarray的形式返回 df.columns列标签 df.index行索引


```python
import pandas as pd
import numpy as np
```


```python
# Series
```


```python
obj = pd.Series([1,2,3,4])
```


```python
obj
```




    0    1
    1    2
    2    3
    3    4
    dtype: int64




```python
obj.values
```




    array([1, 2, 3, 4], dtype=int64)




```python
obj.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
obj2 = pd.Series([1,2,3,4], index=['a','b','c','d'])
```


```python
obj2
```




    a    1
    b    2
    c    3
    d    4
    dtype: int64




```python
obj2.index
```




    Index(['a', 'b', 'c', 'd'], dtype='object')




```python
obj2['a']
```




    1




```python
obj2[['a','b']]
```




    a    1
    b    2
    dtype: int64




```python
obj2[obj2>2]
```




    c    3
    d    4
    dtype: int64




```python
np.exp(obj2)
```




    a     2.718282
    b     7.389056
    c    20.085537
    d    54.598150
    dtype: float64




```python
'b' in obj2
```




    True




```python
'e' in obj2
```




    False




```python
sdata = {'a':1,'b':2,'c':3,'d':5}
```


```python
obj3 = pd.Series(sdata)
```


```python
obj3
```




    a    1
    b    2
    c    3
    d    5
    dtype: int64




```python
obj4 = pd.Series(sdata, index=['c','d','e'])
```


```python
obj4
```




    c    3.0
    d    5.0
    e    NaN
    dtype: float64




```python
pd.isnull(obj4)
```




    c    False
    d    False
    e     True
    dtype: bool




```python
pd.notnull(obj4)
```




    c     True
    d     True
    e    False
    dtype: bool




```python
obj4.isnull()
```




    c    False
    d    False
    e     True
    dtype: bool




```python
obj3
```




    a    1
    b    2
    c    3
    d    5
    dtype: int64




```python
obj4
```




    c    3.0
    d    5.0
    e    NaN
    dtype: float64




```python
obj3+obj4
```




    a     NaN
    b     NaN
    c     6.0
    d    10.0
    e     NaN
    dtype: float64




```python
obj4.name = 'pop'
obj4.index.name = 'bob'
```


```python
obj4
```




    bob
    c    3.0
    d    5.0
    e    NaN
    Name: pop, dtype: float64




```python
obj4.index=['ss','xx','ww']
```


```python
obj4
```




    ss    3.0
    xx    5.0
    ww    NaN
    Name: pop, dtype: float64




```python
# DataFrame
```


```python
data = {'a':[1,2,3,4],'b':[2,2,3,4],'c':[3,2,3,4]}
```


```python
frame = pd.DataFrame(data)
```


```python
frame
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.head()
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(data,columns=['c','b','a'])
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
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame['a'] # 选取列
```




    0    1
    1    2
    2    3
    3    4
    Name: a, dtype: int64




```python
frame.loc[3] # 选取行
```




    a    4
    b    4
    c    4
    Name: 3, dtype: int64




```python
frame.iloc[3] # 选取行
```




    a    4
    b    4
    c    4
    Name: 3, dtype: int64




```python
del frame['c']
```


```python
frame
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
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop = {'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
```


```python
frame3 = pd.DataFrame(pop)
```


```python
frame3
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
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame3.T
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
      <th>2001</th>
      <th>2002</th>
      <th>2000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nevada</th>
      <td>2.4</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>1.7</td>
      <td>3.6</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame3.values
```




    array([[2.4, 1.7],
           [2.9, 3.6],
           [nan, 1.5]])




```python
frame3.columns
```




    Index(['Nevada', 'Ohio'], dtype='object')




```python
frame3.index
```




    Int64Index([2001, 2002, 2000], dtype='int64')



## 2、基本功能

* reindex可以重建索引
* drop可以轴向上删除条目
* 索引、选择与过滤
* ser[:1] 整数索引 要使用标签索引 用loc 整数索引最好显示指明iloc 如果你有一个包含整数的轴索引，数据选择时请始终使用标签索引
* 算术时会自动对齐，类似于索引标签的自动外连接, DataFrame之间行和列都会对齐，没有交叠的标签位置上，内部数据对齐会产生缺失值，可以使用add等运算方法，指定fill_value关键字参数，将缺失值填充为给定的fill_value
* DataFrame和Series间运算时会采用广播机制，默认情况下，会将Series的索引和DataFrame的列进行匹配，并广播到各行，如果一个索引值不在DataFrame的列中，或不在Series的索引中，则对象会重建索引并形成联合，如果想改为在列上进行广播，在行上匹配，则必须使用算法方法中的一种，并指定axis='index'或者axis=0
* numpy的通用函数（逐元素数组方法）对pandas对象也有效
* 另一个常用的操作是DataFrame对象的apply方法，它将函数应用到每一行或每一列的一维数组上，默认对每一列调用一次函数，要对每一行调用，可以传递axis='columns'或axis=1，传递给apply的函数并不一定要返回一个标量，也可以返回带有多个值的Series
* 大部分最常用的数组统计(如sum和mean等)都是DataFrame的方法，因此计算统计值时使用apply并不是必需的
* DataFrame对象的applymap方法以及Series对象的map方法可以将一个函数应用到其每个元素上，即逐元素调用函数
* sort_index方法可以按行或列索引进行字典型排序，sort_values可以根据值按列进行排序，sort_values的可选参数by可以指定一列或多列作为排序键
* rank可以实现Series排名，以及对DataFrame指定axis的排名，即按行或按列，默认按列axis=0
* 带有重复标签的轴索引情况下，来自索引的输出类型可能因标签是否重复而有所不同，但所有的都会选择出来


```python
obj = pd.Series([4.5, 7.2, -5.3, 3.6],index=['d','b','a','c'])
```


```python
obj
```




    d    4.5
    b    7.2
    a   -5.3
    c    3.6
    dtype: float64




```python
obj2 = obj.reindex(['a','b','c','d','e'])
```


```python
obj2
```




    a   -5.3
    b    7.2
    c    3.6
    d    4.5
    e    NaN
    dtype: float64




```python
data = pd.DataFrame(np.arange(16).reshape((4,4)), index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])
```


```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['Colorado','Ohio'])
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop('two',axis=1)
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
      <th>one</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['two','four'], axis='columns')
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
      <th>one</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['two','four'], axis='columns',inplace=True)
```


```python
data
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
      <th>one</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 索引、选择与过滤
```


```python
obj = pd.Series(np.arange(4.), index=['a','b','c','d'])
```


```python
obj
```




    a    0.0
    b    1.0
    c    2.0
    d    3.0
    dtype: float64




```python
obj['b']
```




    1.0




```python
obj[1]
```




    1.0




```python
obj[:3]
```




    a    0.0
    b    1.0
    c    2.0
    dtype: float64




```python
obj[['b','a','d']]
```




    b    1.0
    a    0.0
    d    3.0
    dtype: float64




```python
obj[[1,3]]
```




    b    1.0
    d    3.0
    dtype: float64




```python
obj[obj<2]
```




    a    0.0
    b    1.0
    dtype: float64




```python
obj['b':'c'] # 含尾部，和2:4这种切片不同
```




    b    1.0
    c    2.0
    dtype: float64




```python
data = pd.DataFrame(np.arange(16).reshape((4,4)), index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])
```


```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['two']
```




    Ohio         1
    Colorado     5
    Utah         9
    New York    13
    Name: two, dtype: int32




```python
data[['three','one']]
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
      <th>three</th>
      <th>one</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>10</td>
      <td>8</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>14</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[:2]
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data['three']>5]
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data<5]=0
```


```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc['Colorado',['two','three']]
```




    two      5
    three    6
    Name: Colorado, dtype: int32




```python
data.iloc[2,[3,0,1]]
```




    four    11
    one      8
    two      9
    Name: Utah, dtype: int32




```python
ser = pd.Series(np.arange(3.),index=['a','b','c'])
```


```python
ser[-1]
```




    2.0




```python
ser = pd.Series(np.arange(3.))
```


```python
ser[:1]
```




    0    0.0
    dtype: float64




```python
ser.iloc[:1]
```




    0    0.0
    dtype: float64




```python
ser.loc[:1]
```




    0    0.0
    1    1.0
    dtype: float64




```python
# 算术和数据对齐
```


```python
df1 = pd.DataFrame({'A':[1,2]})
```


```python
df2 = pd.DataFrame({'B':[3,4]})
```


```python
df1+df2
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame = pd.DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index=list('abcd'))
```


```python
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
series = frame.iloc[0]
```


```python
series
```




    b    0.0
    d    1.0
    e    2.0
    Name: a, dtype: float64




```python
frame - series
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
series3 = frame['d']
```


```python
series3
```




    a     1.0
    b     4.0
    c     7.0
    d    10.0
    Name: d, dtype: float64




```python
frame.sub(series3,axis='index')
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 函数应用和映射
```


```python
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.sqrt(frame)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.732051</td>
      <td>2.000000</td>
      <td>2.236068</td>
    </tr>
    <tr>
      <th>c</th>
      <td>2.449490</td>
      <td>2.645751</td>
      <td>2.828427</td>
    </tr>
    <tr>
      <th>d</th>
      <td>3.000000</td>
      <td>3.162278</td>
      <td>3.316625</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
f = lambda x: x.max()-x.min()
```


```python
frame.apply(f)
```




    b    9.0
    d    9.0
    e    9.0
    dtype: float64




```python
frame.apply(f,axis='columns')
```




    a    2.0
    b    2.0
    c    2.0
    d    2.0
    dtype: float64




```python
frame.apply(f,axis=1)
```




    a    2.0
    b    2.0
    c    2.0
    d    2.0
    dtype: float64




```python
def f(x):
    return pd.Series([x.min(),x.max()],index=['min','max'])
```


```python
frame.apply(f)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
format = lambda x: '%.2f' % x
```


```python
frame.applymap(format)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.00</td>
      <td>4.00</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6.00</td>
      <td>7.00</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.00</td>
      <td>10.00</td>
      <td>11.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame['e'].map(format)
```




    a     2.00
    b     5.00
    c     8.00
    d    11.00
    Name: e, dtype: object




```python
# 排序
```


```python
obj = pd.Series(range(4), index = ['d','a','b','c'])
```


```python
obj.sort_index()
```




    a    1
    b    2
    c    3
    d    0
    dtype: int64




```python
frame = pd.DataFrame(np.arange(8).reshape((2,4)),index=['three','one'],columns=['d','a','b','c'])
```


```python
frame
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index()
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(axis=1)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>one</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(axis=1,ascending=False)
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
      <th>d</th>
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
obj = pd.Series([4,7,-3,2])
```


```python
obj.sort_values()
```




    2   -3
    3    2
    0    4
    1    7
    dtype: int64




```python
obj = pd.Series([4,np.nan,7,np.nan,-3,2])
```


```python
obj.sort_values()
```




    4   -3.0
    5    2.0
    0    4.0
    2    7.0
    1    NaN
    3    NaN
    dtype: float64




```python
frame = pd.DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
```


```python
frame
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_values(by='b')
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_values(by=['a','b'])
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.rank()
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 含有重复标签的轴索引
```


```python
obj = pd.Series(range(5), index=['a','a','b','b','c'])
```


```python
obj
```




    a    0
    a    1
    b    2
    b    3
    c    4
    dtype: int64




```python
obj.index.is_unique
```




    False




```python
obj['a']
```




    a    0
    a    1
    dtype: int64




```python
obj['c']
```




    4




```python
df = pd.DataFrame(np.random.randn(4,3),index=['a','a','b','b'])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.035984</td>
      <td>0.914446</td>
      <td>-0.679842</td>
    </tr>
    <tr>
      <th>a</th>
      <td>-0.962927</td>
      <td>-1.505593</td>
      <td>-0.644330</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.108516</td>
      <td>-0.599676</td>
      <td>-1.936749</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.401555</td>
      <td>0.465436</td>
      <td>-0.502870</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['b']
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>-1.108516</td>
      <td>-0.599676</td>
      <td>-1.936749</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.401555</td>
      <td>0.465436</td>
      <td>-0.502870</td>
    </tr>
  </tbody>
</table>
</div>



## 3、描述性统计的概述与计算

* pandas对象装配了一个常用数学、统计学方法的集合，其中大部分属于归约或汇总统计的类别，与numpy数组的类似方法相比，它们内建了处理缺失值的功能
* 常用的方法有sum mean max min idxmin idxmax argmin argmax median cumsum cumprod count describe等，大多数可以指定axis以及skipna(默认为True)的值
* Series的corr方法计算的是两个Series中重叠的、非NA的、按索引对齐的值的相关系，相应地，cov计算的是协方差
* DataFrame的corr和cov方法分别以DataFrame的形式返回相关性和协方差矩阵
* DataFrame的corrwith方法，可以计算出DataFrame中的行或列与另一个序列或DataFrame的相关性，该方法传入一个Series时，会返回一个含有为每列计算相关性值的Series，当传入一个DataFrame时，会计算匹配到列名的相关性数值，传入axis='columns'会按行进行计算
* Series的unique方法给出唯一值，value_counts方法计算包含的不同的值各自的个数，isin执行向量化的成员属性检查


```python
df = pd.DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=list('abcd'),columns=['one','two'])
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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>7.10</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.75</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sum()
```




    one    9.25
    two   -5.80
    dtype: float64




```python
df.sum(axis='columns')
```




    a    1.40
    b    2.60
    c    0.00
    d   -0.55
    dtype: float64




```python
df.mean(axis='columns',skipna=False)
```




    a      NaN
    b    1.300
    c      NaN
    d   -0.275
    dtype: float64




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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.083333</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.493685</td>
      <td>2.262742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.750000</td>
      <td>-4.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.075000</td>
      <td>-3.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.400000</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.250000</td>
      <td>-2.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.100000</td>
      <td>-1.300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
obj =  pd.Series(['a','a','b','c'] * 4)
```


```python
obj.describe()
```




    count     16
    unique     3
    top        a
    freq       8
    dtype: object




```python
# 相关性和协方差
```


```python
import pandas_datareader.data as web
```


```python
df = web.DataReader('005930', 'naver', start='2019-09-10', end='2019-10-09')
```


```python
df.head()
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-09-10</th>
      <td>47100</td>
      <td>47200</td>
      <td>46550</td>
      <td>47000</td>
      <td>9231792</td>
    </tr>
    <tr>
      <th>2019-09-11</th>
      <td>47300</td>
      <td>47400</td>
      <td>46800</td>
      <td>47150</td>
      <td>16141619</td>
    </tr>
    <tr>
      <th>2019-09-16</th>
      <td>47000</td>
      <td>47100</td>
      <td>46400</td>
      <td>47100</td>
      <td>15550926</td>
    </tr>
    <tr>
      <th>2019-09-17</th>
      <td>47000</td>
      <td>47100</td>
      <td>46800</td>
      <td>46900</td>
      <td>7006280</td>
    </tr>
    <tr>
      <th>2019-09-18</th>
      <td>46900</td>
      <td>47700</td>
      <td>46800</td>
      <td>47700</td>
      <td>10413027</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    Open      object
    High      object
    Low       object
    Close     object
    Volume    object
    dtype: object




```python
df['Open'].astype(np.float64).pct_change()
```




    Date
    2019-09-10         NaN
    2019-09-11    0.004246
    2019-09-16   -0.006342
    2019-09-17    0.000000
    2019-09-18   -0.002128
    2019-09-19    0.024520
    2019-09-20    0.028096
    2019-09-23   -0.003036
    2019-09-24   -0.004061
    2019-09-25    0.003058
    2019-09-26   -0.004065
    2019-09-27   -0.020408
    2019-09-30    0.001042
    2019-10-01    0.017690
    2019-10-02   -0.011247
    2019-10-04   -0.019648
    2019-10-07    0.020042
    2019-10-08   -0.009307
    Name: Open, dtype: float64




```python
df['Open'].astype(np.float64).corr(df['Close'].astype(np.float64))
```




    0.8118491766613316




```python
df = df.apply(lambda x:np.float64(x))
```


```python
df.dtypes
```




    Open      float64
    High      float64
    Low       float64
    Close     float64
    Volume    float64
    dtype: object




```python
df.corr()
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Open</th>
      <td>1.000000</td>
      <td>0.871374</td>
      <td>0.969309</td>
      <td>0.811849</td>
      <td>-0.236663</td>
    </tr>
    <tr>
      <th>High</th>
      <td>0.871374</td>
      <td>1.000000</td>
      <td>0.915294</td>
      <td>0.952594</td>
      <td>-0.138376</td>
    </tr>
    <tr>
      <th>Low</th>
      <td>0.969309</td>
      <td>0.915294</td>
      <td>1.000000</td>
      <td>0.891622</td>
      <td>-0.259819</td>
    </tr>
    <tr>
      <th>Close</th>
      <td>0.811849</td>
      <td>0.952594</td>
      <td>0.891622</td>
      <td>1.000000</td>
      <td>-0.045157</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>-0.236663</td>
      <td>-0.138376</td>
      <td>-0.259819</td>
      <td>-0.045157</td>
      <td>1.000000</td>
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Open</th>
      <td>7.491830e+05</td>
      <td>6.731373e+05</td>
      <td>7.625327e+05</td>
      <td>6.383497e+05</td>
      <td>-7.187653e+08</td>
    </tr>
    <tr>
      <th>High</th>
      <td>6.731373e+05</td>
      <td>7.965441e+05</td>
      <td>7.424510e+05</td>
      <td>7.723284e+05</td>
      <td>-4.333389e+08</td>
    </tr>
    <tr>
      <th>Low</th>
      <td>7.625327e+05</td>
      <td>7.424510e+05</td>
      <td>8.260458e+05</td>
      <td>7.361601e+05</td>
      <td>-8.285834e+08</td>
    </tr>
    <tr>
      <th>Close</th>
      <td>6.383497e+05</td>
      <td>7.723284e+05</td>
      <td>7.361601e+05</td>
      <td>8.252369e+05</td>
      <td>-1.439400e+08</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>-7.187653e+08</td>
      <td>-4.333389e+08</td>
      <td>-8.285834e+08</td>
      <td>-1.439400e+08</td>
      <td>1.231194e+13</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.corrwith(df.Open)
```




    Open      1.000000
    High      0.871374
    Low       0.969309
    Close     0.811849
    Volume   -0.236663
    dtype: float64




```python
df[['Open','Close']].corrwith(df[['High','Low']])
```




    Close   NaN
    High    NaN
    Low     NaN
    Open    NaN
    dtype: float64




```python
df.corrwith(df)
```




    Open      1.0
    High      1.0
    Low       1.0
    Close     1.0
    Volume    1.0
    dtype: float64




```python
# 唯一值、计数和成员属性
```


```python
obj = pd.Series(['c','c','a','a','b'])
```


```python
obj
```




    0    c
    1    c
    2    a
    3    a
    4    b
    dtype: object




```python
obj.unique()
```




    array(['c', 'a', 'b'], dtype=object)




```python
obj.value_counts()
```




    c    2
    a    2
    b    1
    dtype: int64




```python
pd.value_counts(obj.values, sort=False)
```




    c    2
    a    2
    b    1
    dtype: int64




```python
mask = obj.isin(['b','c'])
```


```python
mask
```




    0     True
    1     True
    2    False
    3    False
    4     True
    dtype: bool




```python
obj[mask]
```




    0    c
    1    c
    4    b
    dtype: object




```python
to_match = pd.Series(['c','a','b','b','c','a'])
```


```python
unique_vals = pd.Series(['c','b','a'])
```


```python
pd.Index(unique_vals).get_indexer(to_match)
```




    array([0, 2, 1, 1, 0, 2], dtype=int64)




```python
data = pd.DataFrame({'Qu1':[1,3,4,3,4],'Qu2':[2,3,1,2,3],'Qu3':[1,5,2,4,4]})
```


```python
data
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
      <th>Qu1</th>
      <th>Qu2</th>
      <th>Qu3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = data.apply(pd.value_counts).fillna(0)
```


```python
result
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
      <th>Qu1</th>
      <th>Qu2</th>
      <th>Qu3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## 4、数据载入、存储及文件格式

* 输入与输出通常有以下几种类型：
    * 读取文本文件及硬盘上其他更高效的格式文件
    * 从数据库载入数据
    * 与网络资源进行交互(如web api)
* 最常用的几个：read_csv(逗号是默认分隔符), read_table(制表符'\t'是默认分隔符), read_excel, read_html, read_sql, read_json, to_csv, to_excel, to_json等


```python
!type examples\\ex1.csv
```

    a,b,c,d,message
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    


```python
df = pd.read_csv('./examples/ex1.csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_table('examples/ex1.csv', sep=',')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\\ex2.csv
```

    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    


```python
pd.read_csv('examples/ex2.csv', header=None)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv('examples/ex2.csv', names=['a','b','c','d','message'])
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv('examples/ex2.csv', names=['a','b','c','d','message'], index_col='message')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>message</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hello</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>world</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\\csv_mindex.csv
```

    key1,key2,value1,value2
    one,a,1,2
    one,b,3,4
    one,c,5,6
    one,d,7,8
    two,a,9,10
    two,b,11,12
    two,c,13,14
    two,d,15,16
    


```python
parsed = pd.read_csv('examples/csv_mindex.csv',index_col=['key1','key2'])
```


```python
parsed # 分层索引
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
      <th>value1</th>
      <th>value2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">one</th>
      <th>a</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>c</th>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>a</th>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>c</th>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>d</th>
      <td>15</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(open('examples/ex3.txt'))
```




    ['            A         B         C\n',
     'aaa -0.264438 -1.026059 -0.619500\n',
     'bbb  0.927272  0.302904 -0.032399\n',
     'ccc -0.264273 -0.386314 -0.217601\n',
     'ddd -0.871858 -0.348382  1.100491\n']




```python
result = pd.read_table('examples/ex3.txt', sep='\s+')
```


```python
result
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aaa</th>
      <td>-0.264438</td>
      <td>-1.026059</td>
      <td>-0.619500</td>
    </tr>
    <tr>
      <th>bbb</th>
      <td>0.927272</td>
      <td>0.302904</td>
      <td>-0.032399</td>
    </tr>
    <tr>
      <th>ccc</th>
      <td>-0.264273</td>
      <td>-0.386314</td>
      <td>-0.217601</td>
    </tr>
    <tr>
      <th>ddd</th>
      <td>-0.871858</td>
      <td>-0.348382</td>
      <td>1.100491</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\\ex4.csv
```

    # hey!
    a,b,c,d,message
    # just wanted to make things more difficult for you
    # who reads CSV files with computers, anyway?
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    


```python
pd.read_csv('examples/ex4.csv', skiprows=[0,2,3])
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type examples\\ex5.csv
```

    something,a,b,c,d,message
    one,1,2,3,4,NA
    two,5,6,,8,world
    three,9,10,11,12,foo
    


```python
result = pd.read_csv('examples/ex5.csv')
```


```python
result
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.isnull(result)
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.read_csv('examples/ex5.csv', na_values=['NULL'])
```


```python
result
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentinels = {'message': ['foo','NA'], 'something': ['two']}
```


```python
pd.read_csv('examples/ex5.csv', na_values=sentinels)
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.options.display.max_rows = 10
```


```python
result = pd.read_csv('examples/ex6.csv')
```


```python
result
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>2.311896</td>
      <td>-0.417070</td>
      <td>-1.409599</td>
      <td>-0.515821</td>
      <td>L</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>-0.479893</td>
      <td>-0.650419</td>
      <td>0.745152</td>
      <td>-0.646038</td>
      <td>E</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0.523331</td>
      <td>0.787112</td>
      <td>0.486066</td>
      <td>1.093156</td>
      <td>K</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>-0.362559</td>
      <td>0.598894</td>
      <td>-1.843201</td>
      <td>0.887292</td>
      <td>G</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>-0.096376</td>
      <td>-1.012999</td>
      <td>-0.657431</td>
      <td>-0.573315</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>




```python
result = pd.read_csv('examples/ex6.csv', nrows=5)
```


```python
result
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
chunker = pd.read_csv('examples/ex6.csv', chunksize=1000)
```


```python
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(),fill_value=0)
```

    C:\Users\11098\AppData\Local\Temp\ipykernel_17488\2481927249.py:1: FutureWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
      tot = pd.Series([])
    


```python
tot.sort_values(ascending=False)
```




    E    368.0
    X    364.0
    L    346.0
    O    343.0
    Q    340.0
         ...  
    5    157.0
    2    152.0
    0    151.0
    9    150.0
    1    146.0
    Length: 36, dtype: float64




```python
data = pd.read_csv('examples/ex5.csv')
```


```python
data
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.to_csv('examples/out.csv')
```


```python
!type examples\\out.csv
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,
    1,two,5,6,,8,world
    2,three,9,10,11.0,12,foo
    


```python
import sys
```


```python
data.to_csv(sys.stdout, sep='|')
```

    |something|a|b|c|d|message
    0|one|1|2|3.0|4|
    1|two|5|6||8|world
    2|three|9|10|11.0|12|foo
    


```python
data.to_csv(sys.stdout, na_rep='NULL')
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,NULL
    1,two,5,6,NULL,8,world
    2,three,9,10,11.0,12,foo
    


```python
data.to_csv(sys.stdout, index=False, header=False)
```

    one,1,2,3.0,4,
    two,5,6,,8,world
    three,9,10,11.0,12,foo
    


```python
data.to_csv(sys.stdout, index=False, columns=['a','b','c'])
```

    a,b,c
    1,2,3.0
    5,6,
    9,10,11.0
    


```python
dates = pd.date_range('1/1/2000', periods=7)
```


```python
ts = pd.Series(np.arange(7), index=dates)
```


```python
ts
```




    2000-01-01    0
    2000-01-02    1
    2000-01-03    2
    2000-01-04    3
    2000-01-05    4
    2000-01-06    5
    2000-01-07    6
    Freq: D, dtype: int32




```python
ts.to_csv('examples/tseries.csv',header=None)
```


```python
!type examples\\tseries.csv
```

    2000-01-01,0
    2000-01-02,1
    2000-01-03,2
    2000-01-04,3
    2000-01-05,4
    2000-01-06,5
    2000-01-07,6
    

## 5、数据清洗与准备
* 缺失值处理常用函数isnull notnull dropna fillna
* DataFrame的duplicated方法得到每一行是否存在重复（与之前出现过的行相同）情况，drop_duplicates返回的是DataFrame，内容是duplicated返回数组中为False的部分
* Series的str.lower方法将每个值都转化为小写, Series的map和DataFrame的applymap是一种可以便捷执行按元素转换及其他清洗相关操作的方法
* Series对象的replace方法可以实现替代值，它和str.replace是不同的，str.replace是对字符串进行按元素替代的
* 与Series类似，轴标签也有map方法，此外还可以通过rename方法重命名轴索引
* pd.cut方法可以用来分箱
* DataFrame或者Series的sample方法可以随机抽样（默认无放回，设置replace为True则有放回），np.random.permutation(N)配合iloc或者等价的take函数可以实现DataFrame的行置换
* pandas的get_dummies函数可以实现哑变量/One-hot编码
* 字符串操作、正则表达式、pandas中的向量化字符串函数


```python
# 缺失值处理
```


```python
string_data = pd.Series([None,'a','b',np.nan])
```


```python
string_data.isnull()
```




    0     True
    1    False
    2    False
    3     True
    dtype: bool




```python
data = pd.Series([1,np.nan,3.5,np.nan,7])
```


```python
data.dropna()
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
data[data.notnull()]
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
data = pd.DataFrame([[1,6.5,3],[1,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,6.5,3]])
```


```python
cleaned = data.dropna()
```


```python
data
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cleaned
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna(how='all')
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[4] = np.nan
```


```python
data
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
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna(axis=1, how='all')
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame(np.random.randn(7,3))
```


```python
df.iloc[:4,1]=np.nan
df.iloc[:2,2]=np.nan
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>NaN</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>NaN</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=2)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>NaN</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>NaN</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>NaN</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>NaN</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(0)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>0.000000</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>0.000000</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna({1:0.5,2:0})
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>0.500000</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>0.500000</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(0,inplace=True)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>0.000000</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>0.000000</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:4,1]=np.nan
df.iloc[:2,2]=np.nan
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>NaN</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>NaN</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill')
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>NaN</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>NaN</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='bfill')
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>0.293811</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>0.293811</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>0.293811</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>0.293811</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='bfill',limit=2)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.787746</td>
      <td>NaN</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.100368</td>
      <td>NaN</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.745234</td>
      <td>0.293811</td>
      <td>-0.049411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.068113</td>
      <td>0.293811</td>
      <td>-0.919076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.391133</td>
      <td>0.293811</td>
      <td>-1.278555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.303593</td>
      <td>-0.319857</td>
      <td>0.433944</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.554998</td>
      <td>-0.216288</td>
      <td>-2.313839</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 数据转换
```


```python
data = pd.DataFrame({'k1':['one','two']*3+['two'],'k2':[1,1,2,3,3,4,4]})
```


```python
data
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.duplicated()
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool




```python
data.drop_duplicates()
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['v1']=range(7)
```


```python
data.drop_duplicates(['k1'])
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop_duplicates(['k1','k2'], keep='last')
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.Series([1,-999,2,-999,-1000,3])
```


```python
data
```




    0       1
    1    -999
    2       2
    3    -999
    4   -1000
    5       3
    dtype: int64




```python
data.replace(-999,np.nan)
```




    0       1.0
    1       NaN
    2       2.0
    3       NaN
    4   -1000.0
    5       3.0
    dtype: float64




```python
data.replace([-999,-1000],np.nan)
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    NaN
    5    3.0
    dtype: float64




```python
data.replace([-999,-1000],[np.nan,0])
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64




```python
data.replace({-999:np.nan,-1000:0})
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64




```python
# 重命名轴索引
```


```python
data = pd.DataFrame(np.arange(12).reshape((3,4)),index=['Ohio', 'Colorado', 'New York'],columns=['one','two','three','four'])
```


```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
transform = lambda x: x[:4].upper()
```


```python
data.index.map(transform)
```




    Index(['OHIO', 'COLO', 'NEW '], dtype='object')




```python
data.index = data.index.map(transform)
```


```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OHIO</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index=str.title, columns=str.upper)
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
      <th>ONE</th>
      <th>TWO</th>
      <th>THREE</th>
      <th>FOUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colo</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OHIO</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index={'OHIO':'INDIANA'},columns={'three':'peekaboo'})
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
      <th>one</th>
      <th>two</th>
      <th>peekaboo</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDIANA</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OHIO</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index={'OHIO':'INDIANA'},columns={'three':'peekaboo'},inplace=True)
```


```python
data
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
      <th>one</th>
      <th>two</th>
      <th>peekaboo</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDIANA</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 离散化和分箱
```


```python
ages = [20,22,25,27,21,23,37,31,61,45,41,32]
```


```python
bins = [18,25,35,60,100]
```


```python
cats = pd.cut(ages, bins)
```


```python
cats
```




    [(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
    Length: 12
    Categories (4, interval[int64, right]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]




```python
cats.codes
```




    array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)




```python
cats.categories
```




    IntervalIndex([(18, 25], (25, 35], (35, 60], (60, 100]], dtype='interval[int64, right]')




```python
pd.value_counts(cats)
```




    (18, 25]     5
    (25, 35]     3
    (35, 60]     3
    (60, 100]    1
    dtype: int64




```python
pd.cut(ages,[18,26,36,61,100],right=False)
```




    [[18, 26), [18, 26), [18, 26), [26, 36), [18, 26), ..., [26, 36), [61, 100), [36, 61), [36, 61), [26, 36)]
    Length: 12
    Categories (4, interval[int64, left]): [[18, 26) < [26, 36) < [36, 61) < [61, 100)]




```python
group_names = ['Youth','YoungAdult','MiddleAged','Senior']
```


```python
pd.cut(ages,bins,labels=group_names)
```




    ['Youth', 'Youth', 'Youth', 'YoungAdult', 'Youth', ..., 'YoungAdult', 'Senior', 'MiddleAged', 'MiddleAged', 'YoungAdult']
    Length: 12
    Categories (4, object): ['Youth' < 'YoungAdult' < 'MiddleAged' < 'Senior']




```python
data = np.random.rand(20)
```


```python
cats2 = pd.cut(data,4,precision=2)
```


```python
pd.value_counts(cats2)
```




    (0.64, 0.82]    9
    (0.29, 0.47]    6
    (0.11, 0.29]    3
    (0.47, 0.64]    2
    dtype: int64




```python
data = np.random.randn(1000)
```


```python
cats = pd.qcut(data,4)
```


```python
cats
```




    [(0.609, 2.661], (0.609, 2.661], (-2.741, -0.689], (-0.689, -0.0136], (0.609, 2.661], ..., (0.609, 2.661], (-2.741, -0.689], (-0.0136, 0.609], (-0.689, -0.0136], (-2.741, -0.689]]
    Length: 1000
    Categories (4, interval[float64, right]): [(-2.741, -0.689] < (-0.689, -0.0136] < (-0.0136, 0.609] < (0.609, 2.661]]




```python
pd.value_counts(cats)
```




    (-2.741, -0.689]     250
    (-0.689, -0.0136]    250
    (-0.0136, 0.609]     250
    (0.609, 2.661]       250
    dtype: int64




```python
pd.cut(data,[0,0.1,0.5,0.9,1])
```




    [(0.9, 1.0], (0.5, 0.9], NaN, NaN, (0.9, 1.0], ..., (0.9, 1.0], NaN, (0.1, 0.5], NaN, NaN]
    Length: 1000
    Categories (4, interval[float64, right]): [(0.0, 0.1] < (0.1, 0.5] < (0.5, 0.9] < (0.9, 1.0]]




```python
# 置换和随机抽样
```


```python
df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
```


```python
sampler = np.random.permutation(5)
```


```python
sampler
```




    array([1, 4, 0, 2, 3])




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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.take(sampler)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[sampler]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(n=3) # 无放回
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(n=3,replace=True) # 有放回
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 哑变量(One-hot编码)
```


```python
df = pd.DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})
```


```python
pd.get_dummies(df['key'])
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummies = pd.get_dummies(df['key'], prefix='key')
```


```python
df_with_dummy = df[['data1']].join(dummies)
```


```python
df_with_dummy
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
      <th>data1</th>
      <th>key_a</th>
      <th>key_b</th>
      <th>key_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 字符串操作
```


```python
val = 'a,b, guido'
```


```python
val.split(',')
```




    ['a', 'b', ' guido']




```python
pieces = [x.strip() for x in val.split(',')]
```


```python
pieces
```




    ['a', 'b', 'guido']




```python
first, second, third = pieces
```


```python
first+'::'+second+'::'+third
```




    'a::b::guido'




```python
'::'.join(pieces)
```




    'a::b::guido'




```python
'guido' in val
```




    True




```python
val.index(',')
```




    1




```python
val.find(':')
```




    -1




```python
val.count(',')
```




    2




```python
val.replace(',','::')
```




    'a::b:: guido'




```python
# 正则表达式
```


```python
import re
```


```python
text = 'foo    bar\t baz    \tqux'
```


```python
re.split('\s+', text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
regex = re.compile('\s+')
```


```python
regex.split(text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
regex.findall(text)
```




    ['    ', '\t ', '    \t']




```python
# pandas中的向量化字符串
```


```python
data = {'Dave': 'dave@google.com','Steve':'steve@gmail.com','Rob':'rob@gmail.com','Wes':np.nan}
```


```python
data = pd.Series(data)
```


```python
data
```




    Dave     dave@google.com
    Steve    steve@gmail.com
    Rob        rob@gmail.com
    Wes                  NaN
    dtype: object




```python
data.isnull()
```




    Dave     False
    Steve    False
    Rob      False
    Wes       True
    dtype: bool




```python
# 可以用data.map()将字符串和有效的正则表达式方法应用到每个值上，但在NA值上会失败
```


```python
# 为了解决这个问题，Series的str属性有面向数组的方法用于跳过NA值的字符串操作
```


```python
data.str.contains('gmail')
```




    Dave     False
    Steve     True
    Rob       True
    Wes        NaN
    dtype: object




```python
data.str[:5]
```




    Dave     dave@
    Steve    steve
    Rob      rob@g
    Wes        NaN
    dtype: object



## 6、数据规整：连接、联合与重塑

* 在很多应用中，数据可能分布在多个文件或数据库中，抑或以某种不易于分析的格式进行排列
* 分层索引是pandas的重要特性，允许你在一个轴上拥有多个索引层级，swaplevel可以交换层级，sort_index也可以指定level(层级)排序，DataFrame和Series的很多描述性和汇总性统计如sum mean等也有一个level选项，可以对指定层级进行汇总统计，set_index函数可以实现使用DataFrame的列进行索引，reset_index是set_index的反操作
* 数据库风格的DataFrame连接：pd.merge默认内连接，修改how='left'/'right'/'outer'实现左外、右外、全外连接，on/left_on/right_on表示连接的键/左表连接的键/右表连接的键,left_index=True或right_index=True表示左表或者右表根据索引连接, 安装索引连接还有一个方便的join方法
* numpy的concatenate函数可以实现在numpy数组上的沿轴向连接,pandas的concat函数也可以实现Series或DataFrame的沿轴向连接
* combine_first根据传入的对象来修补调用对象的缺失值
* 重排列表格型数据有多种基础操作，这些操作被称为重塑或透视 stack将列的数据透视到行，unstack是stack的反向操作
* pivot和melt分别将“长”透视为“宽”和将“宽”透视为“长”


```python
# 分层索引
```


```python
data = pd.Series(np.random.randn(9),index=[['a','a','a','b','b','c','c','d','d'],[1,2,3,1,3,1,2,2,3]])
```


```python
data
```




    a  1    0.881396
       2   -0.505849
       3    0.242960
    b  1    0.205034
       3    1.058334
    c  1   -1.615628
       2   -0.250232
    d  2   -0.096579
       3    0.366973
    dtype: float64




```python
data.index
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('a', 3),
                ('b', 1),
                ('b', 3),
                ('c', 1),
                ('c', 2),
                ('d', 2),
                ('d', 3)],
               )




```python
data['b']
```




    1    0.205034
    3    1.058334
    dtype: float64




```python
data['b':'c']
```




    b  1    0.205034
       3    1.058334
    c  1   -1.615628
       2   -0.250232
    dtype: float64




```python
data.loc[['b','d']]
```




    b  1    0.205034
       3    1.058334
    d  2   -0.096579
       3    0.366973
    dtype: float64




```python
data.loc[:,2]
```




    a   -0.505849
    c   -0.250232
    d   -0.096579
    dtype: float64




```python
data.unstack()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.881396</td>
      <td>-0.505849</td>
      <td>0.242960</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.205034</td>
      <td>NaN</td>
      <td>1.058334</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-1.615628</td>
      <td>-0.250232</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-0.096579</td>
      <td>0.366973</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.unstack().stack()
```




    a  1    0.881396
       2   -0.505849
       3    0.242960
    b  1    0.205034
       3    1.058334
    c  1   -1.615628
       2   -0.250232
    d  2   -0.096579
       3    0.366973
    dtype: float64




```python
frame = pd.DataFrame(np.arange(12).reshape((4,3)),index=[['a','a','b','b'],[1,2,1,2]],columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])
```


```python
frame
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
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.index.names = ['key1','key2']
```


```python
frame.columns.names = ['states','color']
```


```python
frame
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
      <th>states</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.swaplevel('key1','key2')
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
      <th>states</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(level=1)
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
      <th>states</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>a</th>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.swaplevel(0,1).sort_index(level=0)
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
      <th>states</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sum(level='key2')
```

    C:\Users\11098\AppData\Local\Temp\ipykernel_10508\2004046222.py:1: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
      frame.sum(level='key2')
    




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
      <th>states</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>14</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame = pd.DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':[0,1,2,0,1,2,3]})
```


```python
frame
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>one</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>two</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2</td>
      <td>two</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1</td>
      <td>two</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2 = frame.set_index(['c','d'])
```


```python
frame2
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
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>c</th>
      <th>d</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>0</th>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>0</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.set_index(['c','d'], drop=False)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>c</th>
      <th>d</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>one</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>0</th>
      <td>3</td>
      <td>4</td>
      <td>two</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
      <td>two</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1</td>
      <td>two</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2.reset_index()
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
      <th>c</th>
      <th>d</th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>one</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>two</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 联合与合并数据集
```


```python
df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})
```


```python
df2 = pd.DataFrame({'key':['a','b','d'],'data2':range(3)})
```


```python
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
      <th>key</th>
      <th>data1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
      <th>key</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1,df2)
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(df1,df2,on='key')
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = pd.DataFrame({'lkey':['b','b','a','c','a','a','b'],'data1':range(7)})
```


```python
df4 = pd.DataFrame({'rkey':['a','b','d'],'data2':range(3)})
```


```python
pd.merge(df3,df4,left_on='lkey',right_on='rkey')
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
      <th>lkey</th>
      <th>data1</th>
      <th>rkey</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>0</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>6</td>
      <td>b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2</td>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>4</td>
      <td>a</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a</td>
      <td>5</td>
      <td>a</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
arr = np.arange(12).reshape((3,4))
```


```python
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
np.concatenate([arr,arr],axis=1)
```




    array([[ 0,  1,  2,  3,  0,  1,  2,  3],
           [ 4,  5,  6,  7,  4,  5,  6,  7],
           [ 8,  9, 10, 11,  8,  9, 10, 11]])




```python
s1 = pd.Series([0,1],index=['a','b'])
```


```python
s2 = pd.Series([2,3,4],index=['c','d','e'])
```


```python
s3 = pd.Series([5,6],index=['f','g'])
```


```python
pd.concat([s1,s2,s3])
```




    a    0
    b    1
    c    2
    d    3
    e    4
    f    5
    g    6
    dtype: int64




```python
pd.concat([s1,s2,s3],axis=1)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>f</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>g</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
s4 = pd.concat([s1,s3])
```


```python
s4
```




    a    0
    b    1
    f    5
    g    6
    dtype: int64




```python
pd.concat([s1,s4],axis=1)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>f</th>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>g</th>
      <td>NaN</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([s1,s4],axis=1,join='inner')
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame({'key':['foo','bar','baz'],'A':[1,2,3],'B':[4,5,6],'C':[7,8,9]})
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
      <th>key</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bar</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>baz</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
melted = pd.melt(df,['key'])
```


```python
melted
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
      <th>key</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bar</td>
      <td>A</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>baz</td>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>foo</td>
      <td>B</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bar</td>
      <td>B</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>baz</td>
      <td>B</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>foo</td>
      <td>C</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bar</td>
      <td>C</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>baz</td>
      <td>C</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
reshaped = melted.pivot('key','variable','value')
```


```python
reshaped
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
      <th>variable</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bar</th>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>baz</th>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
reshaped.reset_index()
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
      <th>variable</th>
      <th>key</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bar</td>
      <td>2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>baz</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>foo</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



## 7、数据聚合与分组操作

* groupby函数可以按照指定的键分组，得到一个GroupBy对象，然后可以调用GroupBy对象的聚合方法，实现分组聚合运算
* GroupBy对象支持迭代，会生成一个包含组名和数据块的2维元组序列
* groupby也可以指定level实现根据索引层级分组
* 使用自定义的聚合函数，需要将函数传递给aggregate或agg方法
* 可以通过向groupby传递as_index=False来禁止分组键作为索引的行为，或者调用reset_index来达到同样的效果
* GroupBy方法最常见的目的是apply, Split-Apply-Combine是GruoupBy机制的核心原理
* 数据透视表是常见的数据汇总工具，pivot_table函数可实现数据透视表，传入参数：要汇总的value名，行名，列名，可选参数margins=True代表添加汇总行和汇总列，默认以均值方式汇总，要使用别的方式，传入函数给aggfunc，此外设置fill_value可指定缺失值的填充值，交叉表是数据透视表的特殊情形，计算的是分组中的频次(即count计数)


```python
df = pd.DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'],'data1':np.random.randn(5),'data2':np.random.randn(5)})
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
      <th>key1</th>
      <th>key2</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>one</td>
      <td>-1.786892</td>
      <td>-0.723834</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>two</td>
      <td>-0.016820</td>
      <td>1.346327</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>one</td>
      <td>1.528298</td>
      <td>0.525090</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>two</td>
      <td>-1.286210</td>
      <td>2.232197</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>one</td>
      <td>-0.228864</td>
      <td>0.240642</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped = df['data1'].groupby(df['key1'])
```


```python
grouped
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x000001C18277CB80>




```python
grouped.mean()
```




    key1
    a   -0.677525
    b    0.121044
    Name: data1, dtype: float64




```python
df.groupby(['key1','key2']).size()
```




    key1  key2
    a     one     2
          two     1
    b     one     1
          two     1
    dtype: int64




```python
for name, group in df.groupby('key1'):
    print(name)
    print(group)
```

    a
      key1 key2     data1     data2
    0    a  one -1.786892 -0.723834
    1    a  two -0.016820  1.346327
    4    a  one -0.228864  0.240642
    b
      key1 key2     data1     data2
    2    b  one  1.528298  0.525090
    3    b  two -1.286210  2.232197
    


```python
for (k1,k2), group in df.groupby(['key1','key2']):
    print((k1,k2))
    print(group)
```

    ('a', 'one')
      key1 key2     data1     data2
    0    a  one -1.786892 -0.723834
    4    a  one -0.228864  0.240642
    ('a', 'two')
      key1 key2    data1     data2
    1    a  two -0.01682  1.346327
    ('b', 'one')
      key1 key2     data1    data2
    2    b  one  1.528298  0.52509
    ('b', 'two')
      key1 key2    data1     data2
    3    b  two -1.28621  2.232197
    


```python
list(df.groupby('key1'))
```




    [('a',
        key1 key2     data1     data2
      0    a  one -1.786892 -0.723834
      1    a  two -0.016820  1.346327
      4    a  one -0.228864  0.240642),
     ('b',
        key1 key2     data1     data2
      2    b  one  1.528298  0.525090
      3    b  two -1.286210  2.232197)]




```python
dict(list(df.groupby('key1')))
```




    {'a':   key1 key2     data1     data2
     0    a  one -1.786892 -0.723834
     1    a  two -0.016820  1.346327
     4    a  one -0.228864  0.240642,
     'b':   key1 key2     data1     data2
     2    b  one  1.528298  0.525090
     3    b  two -1.286210  2.232197}




```python
df.dtypes
```




    key1      object
    key2      object
    data1    float64
    data2    float64
    dtype: object




```python
grouped = df.groupby(df.dtypes,axis=1)
```


```python
for dtype, group in grouped:
    print(dtype)
    print(group)
```

    float64
          data1     data2
    0 -1.786892 -0.723834
    1 -0.016820  1.346327
    2  1.528298  0.525090
    3 -1.286210  2.232197
    4 -0.228864  0.240642
    object
      key1 key2
    0    a  one
    1    a  two
    2    b  one
    3    b  two
    4    a  one
    

## 8、高阶pandas

* 分类数据类型：Categoricl
* Groupy配合transform函数
* 方法链技术：pipe方法


```python
categories = ['foo','bar','baz']
```


```python
codes = [0,1,2,0,0,1]
```


```python
my_cats_2 = pd.Categorical.from_codes(codes,categories)
```


```python
my_cats_2
```




    ['foo', 'bar', 'baz', 'foo', 'foo', 'bar']
    Categories (3, object): ['foo', 'bar', 'baz']




```python
ordered_cat = pd.Categorical.from_codes(codes,categories,ordered=True)
```


```python
ordered_cat
```




    ['foo', 'bar', 'baz', 'foo', 'foo', 'bar']
    Categories (3, object): ['foo' < 'bar' < 'baz']




```python
df = pd.DataFrame({'key':['a','b','c']*4,'value':np.arange(12.)})
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
      <th>key</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>a</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>b</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = df.groupby('key').value
```


```python
for i,s in g:
    print(i)
    print(s)
```

    a
    0    0.0
    3    3.0
    6    6.0
    9    9.0
    Name: value, dtype: float64
    b
    1      1.0
    4      4.0
    7      7.0
    10    10.0
    Name: value, dtype: float64
    c
    2      2.0
    5      5.0
    8      8.0
    11    11.0
    Name: value, dtype: float64
    


```python
g.mean()
```




    key
    a    4.5
    b    5.5
    c    6.5
    Name: value, dtype: float64




```python
g.transform(lambda x: x.mean())
```




    0     4.5
    1     5.5
    2     6.5
    3     4.5
    4     5.5
    5     6.5
    6     4.5
    7     5.5
    8     6.5
    9     4.5
    10    5.5
    11    6.5
    Name: value, dtype: float64




```python
g.transform(lambda x: x.rank(ascending=False))
```




    0     4.0
    1     4.0
    2     4.0
    3     3.0
    4     3.0
    5     3.0
    6     2.0
    7     2.0
    8     2.0
    9     1.0
    10    1.0
    11    1.0
    Name: value, dtype: float64




```python

```
