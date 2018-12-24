## Data description:

The objective is to examine the Chinese Stock market performance in 2018. For this sake, I collected the dataset from the website [iwencai](http://www.iwencai.com).

We have the following key variables.
- `outperform_sse` 
  - %. To which extent they outperform the SSE Composite Index.
- `roe_perc_2016` and `roe_perc_2017`
  -  ROEs for the last two years.
- `bluechips`
  - A simple way of defining blue chip companies is to have consecutively 2 years of ROE greater than 5%. The ROE should also be increasing.
- `industry`
  - The industry of the firm
- `market_capitalization`
  - Market capitalization in unit of billions yuan


The data is cleaned so that I add the following variables for convenience.
 - `min`, `q1`, `q2`, `q3`, `max` represent respectively the min, first, second, third quartile and max.
 - `r0` and `r1` represent the represent respectively the range excluding the outliers.
 - `outlier`: dummy. `is outlier ? 1 : 0`
 - `weight`: market capitalization of the firm divided by market capitalization of the industry
 - `weighted_outperformance`: `weight` * `market_capitalization`

## First try:

### Objective: 

I want to find which industry outperformed the SSE Composite Index. `industry` is a categorical variable, and no natural order exists for this variable. So, in my first try, I grouped the data by industry, and thought about using boxplot to examine to which extent the industry outperform the SSE Composite Index.

概要 - 不超过 4 句，简要介绍你的数据可视化和添加任何有助于读者理解的背景信息
设计 - 解释你的设计选择，包括在收集反馈后对可视化做的更改

### First Version and Feedback.

Features of the first try:
1. I use boxplot to represent the extent to which the stocks outperform the SSE Composite Index by industry.
2. The result is sorted by the median of industrial level performance.

Feedback:
1. What my friend learned from the chart: 
  - the median of all the industries are negative.
  - It seems that industries related to food expenditures and consumption ('食品饮料', '纺织服装', '餐饮旅游') perform better than others and have less variation in returns. 
  - Due to the event of ZTE, the telecommunication industry ('信息服务') has reasonablly bad performance and large variance.
2. Comments and suggestions
  - comments: I understand that drawing boxplot using d3 is tricky, but it seems that one can hardly discover  anything interesting apart from the above observations.
  - suggestions: I suggest that you do not use this complicated boxplot. Maybe you can try the bar chart to compare the average performances among industries. You may want to compare the blue chip firms and the other firms within the same industry.

## Second Try:

### Objective: 

I use bar chart to have a look at which industry has better performance compared to the SSE Composite index.

### Second version and feedback from the friend.

Features of the second try:



Features of the third try:
1. The y-axis represents the percentage of companies which outperform the SSE Composite Index.
2. I use grouped bar charts to compare industry level performance. Given Industry, I compare the performance of blue chips companies and non blue chips companies.
3. Industries are ordered by the intrustrial level market capitalization.
4. I use tooltips to enable audience to look into the exact number by moving mouse to the bars.
5. I add grid.

feedback:
1. Graphically, there is no sign that the industries how you order the industries.
2. For a value 0, the bar will not display, that is why we only have one bar for the category "综合". 
3. The measure "percentage of companies which outperform the SSE Composite Index" is not very informative.

Too many industries are present. Not enough information is transmitted.


Have you noticed any features from this chart? (你在这个可视化中注意到什么？)
Do you have problem with this data? (你对这个数据有什么问题吗？)
Are you able to notice the relationship between the variables? (你是否注意到数据关系？)
What information do you think the author would like to convey?  (你觉得这个可视化主要表达了什么？)
Is there anything unclear about this chart? (这个图形中你有什么不明白的地方吗？)

Second try:

Third try: with interactive lines.



### Resources:
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



