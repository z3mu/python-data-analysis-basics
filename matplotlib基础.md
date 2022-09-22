# 绘图与可视化

## 1、简明matplotlib API入门


```python
import matplotlib.pyplot as plt
import numpy as np
```


```python
data = np.arange(10)
```


```python
data
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
plt.plot(data)
```




    [<matplotlib.lines.Line2D at 0x184ec78d910>]




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_5_1.png)
    



```python
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
plt.plot([1.5,3.5,-2,1.6])
```




    [<matplotlib.lines.Line2D at 0x184ee9ab700>]




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_6_1.png)
    



```python
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
_ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30))
ax3.plot(np.random.randn(50).cumsum(),'k--')
```




    [<matplotlib.lines.Line2D at 0x184eeba0040>]




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_7_1.png)
    



```python
fig, axes = plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)
```


    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_8_0.png)
    



```python
plt.plot?
```


    [1;31mSignature:[0m [0mplt[0m[1;33m.[0m[0mplot[0m[1;33m([0m[1;33m*[0m[0margs[0m[1;33m,[0m [0mscalex[0m[1;33m=[0m[1;32mTrue[0m[1;33m,[0m [0mscaley[0m[1;33m=[0m[1;32mTrue[0m[1;33m,[0m [0mdata[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m [1;33m**[0m[0mkwargs[0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m
    Plot y versus x as lines and/or markers.
    
    Call signatures::
    
        plot([x], y, [fmt], *, data=None, **kwargs)
        plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
    
    The coordinates of the points or line nodes are given by *x*, *y*.
    
    The optional parameter *fmt* is a convenient way for defining basic
    formatting like color, marker and linestyle. It's a shortcut string
    notation described in the *Notes* section below.
    
    >>> plot(x, y)        # plot x and y using default line style and color
    >>> plot(x, y, 'bo')  # plot x and y using blue circle markers
    >>> plot(y)           # plot y using x as index array 0..N-1
    >>> plot(y, 'r+')     # ditto, but with red plusses
    
    You can use `.Line2D` properties as keyword arguments for more
    control on the appearance. Line properties and *fmt* can be mixed.
    The following two calls yield identical results:
    
    >>> plot(x, y, 'go--', linewidth=2, markersize=12)
    >>> plot(x, y, color='green', marker='o', linestyle='dashed',
    ...      linewidth=2, markersize=12)
    
    When conflicting with *fmt*, keyword arguments take precedence.
    
    
    **Plotting labelled data**
    
    There's a convenient way for plotting objects with labelled data (i.e.
    data that can be accessed by index ``obj['y']``). Instead of giving
    the data in *x* and *y*, you can provide the object in the *data*
    parameter and just give the labels for *x* and *y*::
    
    >>> plot('xlabel', 'ylabel', data=obj)
    
    All indexable objects are supported. This could e.g. be a `dict`, a
    `pandas.DataFrame` or a structured numpy array.
    
    
    **Plotting multiple sets of data**
    
    There are various ways to plot multiple sets of data.
    
    - The most straight forward way is just to call `plot` multiple times.
      Example:
    
      >>> plot(x1, y1, 'bo')
      >>> plot(x2, y2, 'go')
    
    - If *x* and/or *y* are 2D arrays a separate data set will be drawn
      for every column. If both *x* and *y* are 2D, they must have the
      same shape. If only one of them is 2D with shape (N, m) the other
      must have length N and will be used for every data set m.
    
      Example:
    
      >>> x = [1, 2, 3]
      >>> y = np.array([[1, 2], [3, 4], [5, 6]])
      >>> plot(x, y)
    
      is equivalent to:
    
      >>> for col in range(y.shape[1]):
      ...     plot(x, y[:, col])
    
    - The third way is to specify multiple sets of *[x]*, *y*, *[fmt]*
      groups::
    
      >>> plot(x1, y1, 'g^', x2, y2, 'g-')
    
      In this case, any additional keyword argument applies to all
      datasets. Also this syntax cannot be combined with the *data*
      parameter.
    
    By default, each line is assigned a different style specified by a
    'style cycle'. The *fmt* and line property parameters are only
    necessary if you want explicit deviations from these defaults.
    Alternatively, you can also change the style cycle using
    :rc:`axes.prop_cycle`.
    
    
    Parameters
    ----------
    x, y : array-like or scalar
        The horizontal / vertical coordinates of the data points.
        *x* values are optional and default to ``range(len(y))``.
    
        Commonly, these parameters are 1D arrays.
    
        They can also be scalars, or two-dimensional (in that case, the
        columns represent separate data sets).
    
        These arguments cannot be passed as keywords.
    
    fmt : str, optional
        A format string, e.g. 'ro' for red circles. See the *Notes*
        section for a full description of the format strings.
    
        Format strings are just an abbreviation for quickly setting
        basic line properties. All of these and more can also be
        controlled by keyword arguments.
    
        This argument cannot be passed as keyword.
    
    data : indexable object, optional
        An object with labelled data. If given, provide the label names to
        plot in *x* and *y*.
    
        .. note::
            Technically there's a slight ambiguity in calls where the
            second label is a valid *fmt*. ``plot('n', 'o', data=obj)``
            could be ``plt(x, y)`` or ``plt(y, fmt)``. In such cases,
            the former interpretation is chosen, but a warning is issued.
            You may suppress the warning by adding an empty format string
            ``plot('n', 'o', '', data=obj)``.
    
    Returns
    -------
    list of `.Line2D`
        A list of lines representing the plotted data.
    
    Other Parameters
    ----------------
    scalex, scaley : bool, default: True
        These parameters determine if the view limits are adapted to the
        data limits. The values are passed on to `autoscale_view`.
    
    **kwargs : `.Line2D` properties, optional
        *kwargs* are used to specify properties like a line label (for
        auto legends), linewidth, antialiasing, marker face color.
        Example::
    
        >>> plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)
        >>> plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')
    
        If you specify multiple lines with one plot call, the kwargs apply
        to all those lines. In case the label object is iterable, each
        element is used as labels for each set of data.
    
        Here is a list of available `.Line2D` properties:
    
        Properties:
        agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array and two offsets from the bottom left corner of the image
        alpha: scalar or None
        animated: bool
        antialiased or aa: bool
        clip_box: `.Bbox`
        clip_on: bool
        clip_path: Patch or (Path, Transform) or None
        color or c: color
        dash_capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}
        dash_joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}
        dashes: sequence of floats (on/off ink in points) or (None, None)
        data: (2, N) array or two 1D arrays
        drawstyle or ds: {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}, default: 'default'
        figure: `.Figure`
        fillstyle: {'full', 'left', 'right', 'bottom', 'top', 'none'}
        gid: str
        in_layout: bool
        label: object
        linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
        linewidth or lw: float
        marker: marker style string, `~.path.Path` or `~.markers.MarkerStyle`
        markeredgecolor or mec: color
        markeredgewidth or mew: float
        markerfacecolor or mfc: color
        markerfacecoloralt or mfcalt: color
        markersize or ms: float
        markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
        path_effects: `.AbstractPathEffect`
        picker: float or callable[[Artist, Event], tuple[bool, dict]]
        pickradius: float
        rasterized: bool
        sketch_params: (scale: float, length: float, randomness: float)
        snap: bool or None
        solid_capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}
        solid_joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}
        transform: unknown
        url: str
        visible: bool
        xdata: 1D array
        ydata: 1D array
        zorder: float
    
    See Also
    --------
    scatter : XY scatter plot with markers of varying size and/or color (
        sometimes also called bubble chart).
    
    Notes
    -----
    **Format Strings**
    
    A format string consists of a part for color, marker and line::
    
        fmt = '[marker][line][color]'
    
    Each of them is optional. If not provided, the value from the style
    cycle is used. Exception: If ``line`` is given, but no ``marker``,
    the data will be a line without markers.
    
    Other combinations such as ``[color][marker][line]`` are also
    supported, but note that their parsing may be ambiguous.
    
    **Markers**
    
    =============   ===============================
    character       description
    =============   ===============================
    ``'.'``         point marker
    ``','``         pixel marker
    ``'o'``         circle marker
    ``'v'``         triangle_down marker
    ``'^'``         triangle_up marker
    ``'<'``         triangle_left marker
    ``'>'``         triangle_right marker
    ``'1'``         tri_down marker
    ``'2'``         tri_up marker
    ``'3'``         tri_left marker
    ``'4'``         tri_right marker
    ``'8'``         octagon marker
    ``'s'``         square marker
    ``'p'``         pentagon marker
    ``'P'``         plus (filled) marker
    ``'*'``         star marker
    ``'h'``         hexagon1 marker
    ``'H'``         hexagon2 marker
    ``'+'``         plus marker
    ``'x'``         x marker
    ``'X'``         x (filled) marker
    ``'D'``         diamond marker
    ``'d'``         thin_diamond marker
    ``'|'``         vline marker
    ``'_'``         hline marker
    =============   ===============================
    
    **Line Styles**
    
    =============    ===============================
    character        description
    =============    ===============================
    ``'-'``          solid line style
    ``'--'``         dashed line style
    ``'-.'``         dash-dot line style
    ``':'``          dotted line style
    =============    ===============================
    
    Example format strings::
    
        'b'    # blue markers with default shape
        'or'   # red circles
        '-g'   # green solid line
        '--'   # dashed line with default color
        '^k:'  # black triangle_up markers connected by a dotted line
    
    **Colors**
    
    The supported color abbreviations are the single letter codes
    
    =============    ===============================
    character        color
    =============    ===============================
    ``'b'``          blue
    ``'g'``          green
    ``'r'``          red
    ``'c'``          cyan
    ``'m'``          magenta
    ``'y'``          yellow
    ``'k'``          black
    ``'w'``          white
    =============    ===============================
    
    and the ``'CN'`` colors that index into the default property cycle.
    
    If the color is the only part of the format string, you can
    additionally use any  `matplotlib.colors` spec, e.g. full names
    (``'green'``) or hex strings (``'#008000'``).
    [1;31mFile:[0m      c:\users\11098\appdata\local\programs\python\python38\lib\site-packages\matplotlib\pyplot.py
    [1;31mType:[0m      function
    



```python
plt.plot(np.random.randn(30).cumsum(),'ko--')
```




    [<matplotlib.lines.Line2D at 0x184ef05bb80>]




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_10_1.png)
    



```python
plt.plot(np.random.randn(30).cumsum(),color='k',linestyle='dashed',marker='o')
```




    [<matplotlib.lines.Line2D at 0x184eed07190>]




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_11_1.png)
    



```python
data = np.random.randn(30).cumsum()
```


```python
plt.plot(data,'k--',label='Default')
plt.plot(data,'k-',drawstyle='steps-post',label='steps-post')
plt.legend(loc='upper left')
```




    <matplotlib.legend.Legend at 0x184f03bd9d0>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_13_1.png)
    



```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0,250,500,750,1000])
labels = ax.set_xticklabels(['one','two','three','four','five'],rotation=30,fontsize='small')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')
```




    Text(0.5, 0, 'Stages')




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_14_1.png)
    



```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0,250,500,750,1000])
labels = ax.set_xticklabels(['one','two','three','four','five'],rotation=30,fontsize='small')
props = {
    'title': 'My first matplotlib plot',
    'xlabel': 'Stages'
}
ax.set(**props)
```




    [Text(0.5, 1.0, 'My first matplotlib plot'), Text(0.5, 0, 'Stages')]




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_15_1.png)
    



```python
from datetime import datetime
import pandas as pd
```


```python
data = pd.read_csv('../pandas基础知识/examples/spx.csv', index_col=0,parse_dates=True)
```


```python
spx = data['SPX']
```


```python
spx
```




    Date
    1990-02-01     328.79
    1990-02-02     330.92
    1990-02-05     331.85
    1990-02-06     329.66
    1990-02-07     333.75
                   ...   
    2011-10-10    1194.89
    2011-10-11    1195.54
    2011-10-12    1207.25
    2011-10-13    1203.66
    2011-10-14    1224.58
    Name: SPX, Length: 5472, dtype: float64




```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
spx.plot(ax=ax,style='k-')
crisis_data = [(datetime(2007,10,11),'Peak of bull market'),(datetime(2008,3,12),'Bear Stearns Fails'),(datetime(2008,9,15),'Lehman Bankruptcy')]
for date, label in crisis_data:
    ax.annotate(label, xy=(date,spx.asof(date)+75),xytext=(date,spx.asof(date)+225),arrowprops=dict(facecolor='black',headwidth=4,width=2,headlength=4),horizontalalignment='left',verticalalignment='top')
   
ax.set_xlim(['1/1/2007','1/1/2011'])
ax.set_ylim([600,1800])
ax.set_title('Import dates in the 2008-2009 financial crisis')
```




    Text(0.5, 1.0, 'Import dates in the 2008-2009 financial crisis')




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_20_1.png)
    



```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rect = plt.Rectangle((0.2,0.75),0.4,0.15,color='k',alpha=0.3)
circ = plt.Circle((0.7,0.2),0.15,color='b',alpha=0.3)
pgon = plt.Polygon([[0.15,0.15],[0.35,0.4],[0.2,0.6]],color='g',alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
plt.savefig('figpath.png',dpi=400,bbox_inches='tight')
```


    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_21_0.png)
    



```python
plt.rc('figure', figsize=(10,10))
```

## 2、使用pandas和seaborn绘图


```python
s = pd.Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
```


```python
s.plot()
```




    <AxesSubplot:>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_25_1.png)
    



```python
s.plot(use_index=False)
```




    <AxesSubplot:>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_26_1.png)
    



```python
df = pd.DataFrame(np.random.randn(10,4).cumsum(0),columns=['A','B','C','D'],index=np.arange(0,100,10))
```


```python
df.plot()
```




    <AxesSubplot:>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_28_1.png)
    



```python
fig, axes = plt.subplots(2,1)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot.bar(ax=axes[0],color='k',alpha=0.7)
data.plot.barh(ax=axes[1],color='k',alpha=0.7)
```




    <AxesSubplot:>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_29_1.png)
    



```python
df = pd.DataFrame(np.random.rand(6,4),index=['one','two','three','four','five','six'],columns=pd.Index(['A','B','C','D'],name='Genus'))
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
      <th>Genus</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>0.794526</td>
      <td>0.523877</td>
      <td>0.637494</td>
      <td>0.113980</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.680581</td>
      <td>0.803333</td>
      <td>0.893919</td>
      <td>0.206183</td>
    </tr>
    <tr>
      <th>three</th>
      <td>0.849330</td>
      <td>0.646299</td>
      <td>0.569287</td>
      <td>0.411957</td>
    </tr>
    <tr>
      <th>four</th>
      <td>0.677244</td>
      <td>0.399135</td>
      <td>0.858010</td>
      <td>0.641172</td>
    </tr>
    <tr>
      <th>five</th>
      <td>0.546959</td>
      <td>0.327479</td>
      <td>0.058202</td>
      <td>0.392600</td>
    </tr>
    <tr>
      <th>six</th>
      <td>0.053830</td>
      <td>0.672359</td>
      <td>0.230896</td>
      <td>0.500954</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.plot.bar()
```




    <AxesSubplot:>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_32_1.png)
    



```python
df.plot.barh(stacked=True, alpha=0.5)
```




    <AxesSubplot:>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_33_1.png)
    



```python
tips = pd.read_csv('../pandas基础知识/examples/tips.csv')
```


```python
tips
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>239</th>
      <td>29.03</td>
      <td>5.92</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>240</th>
      <td>27.18</td>
      <td>2.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>241</th>
      <td>22.67</td>
      <td>2.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>242</th>
      <td>17.82</td>
      <td>1.75</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>243</th>
      <td>18.78</td>
      <td>3.00</td>
      <td>No</td>
      <td>Thur</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 6 columns</p>
</div>




```python
party_counts = pd.crosstab(tips['day'],tips['size'])
```


```python
party_counts
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
      <th>size</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
    <tr>
      <th>day</th>
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
      <th>Fri</th>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>2</td>
      <td>53</td>
      <td>18</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>0</td>
      <td>39</td>
      <td>15</td>
      <td>18</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>1</td>
      <td>48</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
party_counts = party_counts.loc[:,2:5]
```


```python
party_counts
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
      <th>size</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fri</th>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>53</td>
      <td>18</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>39</td>
      <td>15</td>
      <td>18</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>48</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
party_pcts = party_counts.div(party_counts.sum(1),axis=0)
```


```python
party_pcts
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
      <th>size</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fri</th>
      <td>0.888889</td>
      <td>0.055556</td>
      <td>0.055556</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>0.623529</td>
      <td>0.211765</td>
      <td>0.152941</td>
      <td>0.011765</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>0.520000</td>
      <td>0.200000</td>
      <td>0.240000</td>
      <td>0.040000</td>
    </tr>
    <tr>
      <th>Thur</th>
      <td>0.827586</td>
      <td>0.068966</td>
      <td>0.086207</td>
      <td>0.017241</td>
    </tr>
  </tbody>
</table>
</div>




```python
party_pcts.plot.bar()
```




    <AxesSubplot:xlabel='day'>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_42_1.png)
    



```python
import seaborn as sns
```


```python
tips
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>239</th>
      <td>29.03</td>
      <td>5.92</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>240</th>
      <td>27.18</td>
      <td>2.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>241</th>
      <td>22.67</td>
      <td>2.00</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>242</th>
      <td>17.82</td>
      <td>1.75</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>243</th>
      <td>18.78</td>
      <td>3.00</td>
      <td>No</td>
      <td>Thur</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 6 columns</p>
</div>




```python
tips['tip_pct'] = tips['tip']/(tips['total_bill']-tips['tip'])
```


```python
tips.head()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.063204</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.191244</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
      <td>0.199886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
      <td>0.162494</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
      <td>0.172069</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(x='tip_pct',y='day',data=tips,orient='h')
```




    <AxesSubplot:xlabel='tip_pct', ylabel='day'>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_47_1.png)
    



```python
sns.barplot(x='tip_pct',y='day',hue='time',data=tips,orient='h')
sns.set(style='whitegrid')
```


    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_48_0.png)
    



```python
tips['tip_pct'].plot.hist(bins=50)
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_49_1.png)
    



```python
tips['tip_pct'].plot.density()
```




    <AxesSubplot:ylabel='Density'>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_50_1.png)
    



```python
comp1 = np.random.normal(0,1,size=200)
comp2 = np.random.normal(10,2,size=200)
```


```python
values = pd.Series(np.concatenate([comp1,comp2]))
```


```python
sns.distplot(values,bins=100,color='k')
```

    C:\users\11098\appdata\local\programs\python\python38\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:ylabel='Density'>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_53_2.png)
    



```python
macro = pd.read_csv('../pandas基础知识/examples/macrodata.csv')
```


```python
data = macro[['cpi','m1','tbilrate','unemp']]
```


```python
trans_data = np.log(data).diff().dropna()
```


```python
trans_data.tail()
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
      <th>cpi</th>
      <th>m1</th>
      <th>tbilrate</th>
      <th>unemp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>198</th>
      <td>-0.007904</td>
      <td>0.045361</td>
      <td>-0.396881</td>
      <td>0.105361</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-0.021979</td>
      <td>0.066753</td>
      <td>-2.277267</td>
      <td>0.139762</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.002340</td>
      <td>0.010286</td>
      <td>0.606136</td>
      <td>0.160343</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0.008419</td>
      <td>0.037461</td>
      <td>-0.200671</td>
      <td>0.127339</td>
    </tr>
    <tr>
      <th>202</th>
      <td>0.008894</td>
      <td>0.012202</td>
      <td>-0.405465</td>
      <td>0.042560</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.regplot('m1','unemp',data=trans_data)
plt.title('Changes in log %s versus log %s' % ('m1','unemp'))
```




    Text(0.5, 1.0, 'Changes in log m1 versus log unemp')




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_58_1.png)
    



```python
sns.pairplot(trans_data,diag_kind='kde',plot_kws={'alpha':0.2})
```




    <seaborn.axisgrid.PairGrid at 0x184f427b9a0>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_59_1.png)
    



```python
sns.factorplot(x='day',y='tip_pct',hue='time',col='smoker',kind='bar',data=tips[tips.tip_pct<1])
```

    C:\users\11098\appdata\local\programs\python\python38\lib\site-packages\seaborn\categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    




    <seaborn.axisgrid.FacetGrid at 0x1848871ba60>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_60_2.png)
    



```python
sns.factorplot(x='day',y='tip_pct',row='time',col='smoker',kind='bar',data=tips[tips.tip_pct<1])
```

    C:\users\11098\appdata\local\programs\python\python38\lib\site-packages\seaborn\categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    




    <seaborn.axisgrid.FacetGrid at 0x18488a69790>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_61_2.png)
    



```python
sns.factorplot(x='day',y='tip_pct',kind='box',data=tips[tips.tip_pct<1])
```

    C:\users\11098\appdata\local\programs\python\python38\lib\site-packages\seaborn\categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    




    <seaborn.axisgrid.FacetGrid at 0x18488f49d90>




    
![png](matplotlib%E5%9F%BA%E7%A1%80_files/matplotlib%E5%9F%BA%E7%A1%80_62_2.png)
    



```python

```
