---
title: 读书-static-site-generators
date: 2018-01-22 20:57:01
tags: 
    - 读书
categories: 工程开发系列
---


# chapter 1 What are Static sites?

## 1.1 CMS
- CMS: To this day, most of the content published on the Web runs through some form of content Management system. Popular open source options include __Drupal, Joomla, Typo3__. Nowadays, these systems typically handle much more than simply content creation and publication, with features such as complex roles and access control, workflow management, document management, and syndication
### 1.1.1 国内有没有大机构用 Drupal 来构建官网？ https://www.zhihu.com/question/19648357
- Drupal是cms中比较倾向二次开发的一种，国内的一般网站比较习惯于phpcms之类全套都做好，只需要一个管理员之类的。需求不合。
- 从用户体验ue来说，逻辑性强，简洁，所谓less is more。drupal里有view,contenttype,menu,block,region这些概念，就像一块块积木，搭起来以后使页面层层分明。模块只有你想不到的，没有做不到的，还可以在已知模块下进行二次开发。话说，我喜欢在drupal里写js.以上全是个人拙见，
- 学Drupal需要具有良好的php功底，而且英语也得跟上，否则你在国内这些论坛查不出来太多有深度的东西，而且你问个问题经常石沉大海，国内圈子感觉不够活跃。而且大部分都是讲Drupal主题开发，偏前端，讲模块开发的感觉都是浅尝辄止，给你讲个hook然后讲几个api和变量就完事了，没有太好的实例。
-  国内的cms用过动易、supersite、phpcms、dedecms，后来转向drupal。两相比较的话，感觉国内cms在系统架构、代码严谨性、文档完整性、社区支持方面有差距。
- Drupal based Education Sites https://groups.drupal.org/education-sites#China
- dedecms 好比美图秀秀,所见即所得,你可以无任何技术基础就可以通过后台配置搭建一个网站 drupal 好比photoshop ,需要具备一定专业技能你才能做出一个网站.
- dedecms不必说，上不得台面，二次开发时你会发现，最基本的数据库结构和权限控制都是最低级，后期几乎无法扩展的（我们目前在重写整个系统）；Discuz! 算是业界良心产品，
- Chrome Sniffer Plus: chrome 插件 可以解析出网站用的什么技术,可以阅读一下源码
- Wappalyzer 类似的插件
- 为什么国内的中小网站都是用 Discuz! 或者 Phpwind 而不用 WordPress / Joomla / Drupal 呢？  https://www.zhihu.com/question/20813536/answer/36577433
- https://www.zhihu.com/question/23580483/answer/26518536 分析drupal wordpress joomla 三大系统优缺点？

## 1.2 dynamic aspects
### 1.2.1 对静态站点添加动态内容的一些服务
- Disqus, Livefyre, or Facebook for comments
- Wufoo or Google for forms
- Google, Swiftype, or AddSearch for search
- Discourse for forums    
### 1.2.2 BAAS: backend as a service例子  https://www.zhihu.com/question/22098754 这里给出一些BaaS服务提供商
- 后台数据存储
    1. StackMob Product | StackMob
    2. Parse  Products
    3. AVOS Cloud AVOS Cloud
    4. Bmob  Bmob移动后端云服务平台
- 应用数据分析
    1. 友盟 友盟-专业的移动开发者服务平台
    2. TalkingData TalkingData-专业的无线互联网数据服务平台
    3. 魔方  魔方-移动应用服务平台
    4. AVOS Cloud Analytics 功能 - AVOS Cloud
- 移动终端测试
    1.Testin Testin云测
    2.班墨云测试 全球首款智能云测试系统
    3.DroidPilot Android自动化测试工具DroidPilot
    4.摩测  e世博,e世博注册首选平台
- 应用发布
    1.一键云  关于我们
    2.抓猫网 抓猫移动广告聚合优化平台
- 消息推送
    1. 极光推送  JPush极光推送
    2. 个推 个推开放平台
    3. AVOS Cloud Push 功能 - AVOS Cloud
- 信息识别
    1.语义云 首页 (自然语义)
    2.慧眼开发平台 http://smarkeye.mongtx.com/ (图像)
    3.AngelEyes http://www.angeleyes.it/ (图像)
    4.Face++ Face++ 最好的免费人脸识别云服务 (人脸)
    5.Face-API  http://faceapi.cn/ (人脸)
- 应用内广告
    1.掌淘联盟  http://appgo.cn/
    2.抓猫网 抓猫移动广告聚合优化平台
- 数据
    - 聚合数据	专业的数据服务开放平台 https://www.juhe.cn/

BAAS的技术: What makes all of these services work is the ability to load remote
data requests via Ajax.

```javascript
<div id="disqus_thread"></div>
<script type="text/javascript">
    /* * * CONFIGURATION VARIABLES: EDIT BEFORE PASTING INTO
    YOUR WEBPAGE * * */
    var disqus_shortname = 'remotesynthesis'; // required:
    replace example with your forum shortname
    /* * * DON'T EDIT BELOW THIS LINE * * */
    (function() {
    var dsq = document.createElement('script'); dsq.type =
    'text/javascript'; dsq.async = true;
    dsq.src = '//' + disqus_shortname + '.disqus.com/
    embed.js';
    (document.getElementsByTagName('head')[0] || docu
    ment.getElementsByTagName('body')[0]).appendChild(dsq);
    })();
</script>
<noscript>Please enable JavaScript to view the
<a href="https://disqus.com/?ref_noscript"> comments powered by Disqus.</a> </noscript>
```
## 1.3 defining a static website
- 各个解决方干的优缺点
    - static web pages of old failed to meet the needs of the Web as websites became more complex and interactive.
    - Dynamic sites generally and content management systems specifically solved some of these problems but led to increased complexity in both development and authoring.
    - blog engine partially addressed these issues but also took on some of complexity over time.
    - Ajax and the rise of services have helped make static pages a viable option again.
- static site generator的特点
    - statics site files are delivered to the end user exactly as they are on the server.
        - no server-site generation at runtime.
    - no server-side language
    - no database
    - static sites are HTML/CSS/JS
- static sites 的benefits
    - Performance
    - hosting: github pages,surge
    - Security
    - content versioning

# chapter 2 The basics of  static site generator    

## 2.1 what are static site generator    

- Comprehensive Lists of Static Site Generators https://staticsitegenerators.net/    
- template language
    - Liquid
    - Jade
- site examples:   https://github.com/remotesynth/Static-Site-Samples

# chapter 3 popular site generators

- jekyll sites list: https://github.com/jekyll/jekyll/wiki/Sites
