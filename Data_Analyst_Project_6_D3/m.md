### Quetions

### Data description:

The objective is the examine the Chinese Stock market performance on Novmeber 2018.

For this sake, I collected the following dataset from the website [iwencai](http://www.iwencai.com).

We have the following variables variables.
- To which extent they outperform the SSE Composite Index. 
  - `outperform_sse`
- ROEs for the last two years.
  - `roe_perc_2016`
  - `roe_perc_2017`
  - `bluechips`: A simple way of defining bluechip companies is to have consecutively 2 years of roe greater than 5%.
- The industry of the firm
  - `industry`
- Market capitalization in unit of billions yuan
  - `market_capitalization`
  - 

I want to find which industry outperformed the SSE Composite Index, so I grouped the data into different industries.

Firstly, I grouped the data by industry, and thought about using boxplot to examine to which extent the industry outperform the SSE Composite Index.

概要 - 不超过 4 句，简要介绍你的数据可视化和添加任何有助于读者理解的背景信息
设计 - 解释你的设计选择，包括在收集反馈后对可视化做的更改



### Objective: 

I use bar chart to have a look at which industry has better performance compared to the SSE Composite index.

## Feedback

### First version and feedback from the friend.

Features of the first try:
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



