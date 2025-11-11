---
title: 论文阅读 an experimental comparison of click position bias models
date: 2016-08-02
---

论文阅读 an experimental comparison of click position bias models

# introduction

- 统计一下
    
    clickthrough rate decay curve

- 统计一下

    用户回头的概率，examination backward回头在喜马拉雅搜索系统中无法统计，但是click backward回头可以统计。

    代码
    
    图片

- figure 3
    
    从某种方面说明了，当点击概率比较小的时候，比较subfigure1和subfigure9，两者最左端的差异性。即说明在点击概率比较低的时候，position 对点击率的影响占主导地位，排序位置越靠前，position对ctr的影响越大。

- 一般排序性质的评估分为两种
    - manual relevance assessments
    - NDCG(包括gain function 和 discount function)


- 问题
    - click-position bias model 可以用来做什么?怎么用 文献 5 9 3
    - 分类排序，其实本质上是个搜索分类排序问题

- 公共数据集
    - http://jeffhuang.com/search_query_logs.html

- 后续阅读
    - https://www.bing.com/academic/profile?id=1992549066&encoded=0&v=paper_preview&mkt=zh-cn#aca_reference 中的referenced article.
    - Beyond Click Graph: Topic Modeling for Search Engine Query Log Analysis
    - https://www.kaggle.com/c/yandex-personalized-web-search-challenge
    - http://citeseer.ist.psu.edu/viewdoc/download?doi=10.1.1.85.8688&rep=rep1&type=pdf
    - http://www.aclweb.org/anthology/D09-1055
    - http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6664324&url=http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6664324
    - organic & paid search

    - 模型bpr
    - 夹子

```
```


### 资料

- [ Agichtein, Eugene, et al. "Learning user interaction models for predicting web search result preferences." Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 2006. ](http://dl.acm.org/citation.cfm?id=1341545)
- [Baeza‐Yates, Ricardo, Carlos Hurtado, and Marcelo Mendoza. "Improving search engines by query clustering." Journal of the American Society for Information Science and Technology 58.12 (2007): 1793-1804.](http://onlinelibrary.wiley.com/doi/10.1002/asi.20627/full)
- [Dupret, Georges, et al. "A statistical model of query log generation." International Symposium on String Processing and Information Retrieval. Springer Berlin Heidelberg, 2006. ](https://www.researchgate.net/publication/40883559_A_Statistical_Model_of_Query_Log_Generation)
- 3 [Dupret, Georges, Vanessa Murdock, and Benjamin Piwowarski. "Web search engine evaluation using clickthrough data and a user model." WWW2007 workshop Query Log Analysis: Social and Technological Challenges. 2007.](https://www.researchgate.net/publication/228924163_Web_search_engine_evaluation_using_clickthrough_data_and_a_user_model)
- 5 [Joachims, Thorsten. "Optimizing search engines using clickthrough data." Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2002.](https://www.researchgate.net/publication/2556463_Optimizing_Search_Engines_using_Clickthrough_Data)
- [Joachims, Thorsten, et al. "Accurately interpreting clickthrough data as implicit feedback." Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval. Acm, 2005.](https://www.researchgate.net/publication/200110530_Accurately_interpreting_clickthrough_data_as_implicit_feedback)
- 8 [Radlinski, Filip, and Thorsten Joachims. "Minimally invasive randomization for collecting unbiased preferences from clickthrough logs." Proceedings of the National Conference on Artificial Intelligence. Vol. 21. No. 2. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2006.](https://www.researchgate.net/publication/221606968_Minimally_Invasive_Randomization_fro_Collecting_Unbiased_Preferences_from_Clickthrough_Logs)
- 9 [Richardson, Matthew, Ewa Dominowska, and Robert Ragno. "Predicting clicks: estimating the click-through rate for new ads." Proceedings of the 16th international conference on World Wide Web. ACM, 2007.](https://www.researchgate.net/publication/200110656_Predicting_clicks_Estimating_the_click-through_rate_for_new_ads)
- [Baeza‐Yates, Ricardo, Carlos Hurtado, and Marcelo Mendoza. "Improving search engines by query clustering." Journal of the American Society for Information Science and Technology 58.12 (2007): 1793-1804.](https://www.researchgate.net/publication/220435314_Improving_search_engines_by_query_clustering)
