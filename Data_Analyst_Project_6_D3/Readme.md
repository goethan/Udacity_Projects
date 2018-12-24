## Relevant files:

- First try: `Index1.html`
- Second try: `Index2.html`
- Third try: `Index3.html`
- Data: `3.csv`

## Data description:

The objective is to examine the Chinese Stock market performance in 2018. The data is collected from the website [iwencai](http://www.iwencai.com), using the following keywords: `上市天数大于730天,2018年1月1日到2018年12月19日跑赢大盘或2018年1月1日到2018年12月19日跑输大盘,连续2年的roe,2018年12月19日总市值,所属同花顺一级行业,股票,非ST`.

**The term `SSE` is short for Shanghai Stock Exchange.**

We have the following key variables in the data.
- `outperform_sse` 
  - %. To which extent the stock outperforms the SSE Composite Index.
- `roe_perc_2016` and `roe_perc_2017`
  -  ROEs for the last two years.
- `bluechips`
  - A simple way of defining blue chip companies is to have consecutively 2 years of ROE greater than 5%. The ROE should also be increasing.
- `industry`
  - The industry of the firm
- `market_capitalization`
  - Market capitalization in unit of billions yuan


I add the following variables.
 - `min`, `q1`, `q2`, `q3`, `max` represent respectively the min, first, second, third quartile and max.
 - `r0` and `r1` represent respectively the range, excluding the outliers.
 - `outlier`: dummy. `is outlier ? 1 : 0`
 - `weight`: market capitalization of the firm divided by market capitalization of the industry.
 - `weighted_outperformance`: `weight` * `market_capitalization`

## First try:

Objective: I want to examine the stock performance compared to the SSE Composite Index by industry. `industry` is a categorical variable, and there is no natural ordering for this variable. So, in my first try, I grouped the data by industry, and I thought about using *boxplot* to examine to which extent the industry outperforms the SSE Composite Index.

**Features of the first try:**

1. I use boxplot to represent the extent to which the stocks outperform the SSE Composite Index by industry.
2. The industry is sorted by the median of industrial level performance.

**Feedback:**

1. What my friend learned from the chart: 
  - The median of all the industries are negative.
  - It seems that industries related to food expenditures and consumption etc.('食品饮料', '纺织服装', '餐饮旅游') perform better than others and have less variation in returns. 
  - Due to the event of ZTE, the telecommunication industry ('信息服务') has reasonablly bad performance and large variance.
2. Comments and suggestions
  - Comments: I understand that drawing boxplot using d3 is tricky, but it seems that one can hardly discover  anything interesting apart from the above observations.
  - Suggestions: I suggest that you do not use this complicated boxplot. Maybe you can try the bar chart to compare the average performances among industries. You may want to compare the blue chip firms and the other firms within the same industry.

## Second Try:

Objective: I use *bar chart* to look at which industry has better performance on average compared to the SSE Composite index. 

**Features:**

1. Bar chart is used.
2. The y-axis represents the weighted average of industry performance compared to the SSE Index. I distinguish between blue chip and non blue chip firms.
3. For each industry, I compare blue chip and non blue chip firms.

**Feedback:**

1. Comment and Suggestions:
  - Comment: The chart appears to give very noisy information without a consistent pattern. Based on your chart, the audience would not know why you want to distinguish between blue chip and non blue chip firms. 
  - Suggestions: It would be much better if you could focus on something simple. Please try to convey to the audience a clear idea of certain pattern in your chart. For example, I am interested in whether higher fraction of blue chip firms have better performance compared to the non blue chip firms.

## Third Try:

Objective: I use grouped bar chart to examine whether there is higher fraction of blue chip firms which outperform the SSE Index than that of non blue chip firms. 

**Features:**

Features of the third try:

1. The y-axis represents the fraction of companies which outperform the SSE Composite Index.
2. I use grouped bar charts to compare industry level performance. Given Industry, I compare the fraction of firms which outperform the SSE index, and I distinguish between blue chips and non blue chips companies. 
3. Industries are ordered by the intrustrial level market capitalization.
4. I use tooltips to enable audience to look into the exact number by moving mouse to the bars.
5. I add grid.

**Feedback:**
Feedback:
1. Comment and Suggestions:
  - Comment: It is clear from this chart that the fraction of blue chip firms which outperform the SSE index is higher than those non blue chip firms on average.
  - Suggestions: In the future, you may want to include addtional charts to give audience more information, such as the relationship between the market capitalization, and the extent to which the stocks outperform the SSE Index.



## Resources:
1. Negative bar values..
  - https://bl.ocks.org/mbostock/2368837
2. Grouped Bar Chart
  - https://bl.ocks.org/mbostock/3887051
3. launch local server to enable local data loading
  - https://stackoverflow.com/questions/15417437/d3-js-loading-local-data-file-from-file
4. Scaleband
  - https://d3indepth.com/scales/
5. Add grid lines
  - https://bl.ocks.org/d3noob/c506ac45617cf9ed39337f99f8511218
6. Pop text, tooltips.
  - https://stackoverflow.com/questions/10805184/show-data-on-mouseover-of-circle
7. use d3.quantile when grouping variables.
  - https://stackoverflow.com/questions/31705056/d3-key-rollups-using-quantiles
8. boxplot
  - https://beta.observablehq.com/@mbostock/d3-box-plot#margin
9. vertical bars with negative values.
  - https://bl.ocks.org/datafunk/8a17b5f476a40a08ed17



