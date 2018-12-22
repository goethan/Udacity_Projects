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

I use bar graph to compare which industry has better performance compared to the SSE indicator.

elements.. 
grid.

## Feedback
反馈 - 包括从第一份草图到最后定稿，你从他人那里获得的关于你的可视化的所有反馈
### First version and feedback from the friend.

The height of the bar represents the number of the ...
There is no exact number.

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



