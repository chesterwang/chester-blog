---
title: 读书  Front-End Developer Handbook 2017
date: 2018-01-10 01:18:06
tags: 
    - 读书
categories: 工程开发系列
---

# part1 introduction

- 前端runtime
    - web browser
    - headless browser
        - http://blog.csdn.net/u014745198/article/details/60757926
        - A headless browser is a web browser without a graphical user interface.
        - 那么浏览器该有的东西它都应该有，只是看不到界面而已。
        - list of headless browsers: chrome/firefox/PhantomJS/HtmlUnit/TrifleJS/Splash/slimerjs
        - Headless browsers are a web browser without a graphical user interface that can be controlled from a command line interface programmatically for the purpose of web page automation (e.g., functional testing, scraping, unit testing, etc.).
    - WebView
        - Webviews are used by a native OS, in a native application, to run web pages.
        - list
            - Cordova (typically for native phone/tablet apps)
            - NW.js (typically used for desktop apps)
            - Electron (typically used for desktop apps)
    - compilation ouput
        - 用户浏览器上的开发语言来写原生应用
        - list
            - NativeScript
            - React Native
- web technologies employed by front-end developers
    - 核心技术
        - Uniform Resource Locators (aka URLs)
        - Hypertext Transfer Protocol (aka HTTP)
        - Hyper Text Markup Language (aka HTML)
        - Cascading Style Sheets (aka CSS)
        - JavaScript Programming Language (aka ECMAScript 262)
        - JavaScript Object Notation (aka JSON)
        - Document Object Model (aka DOM)
        - Web APIs (aka HTML5 and friends or Browser APIs)
        - Web Content Accessibility Guidelines (aka WCAG) & Accessible Rich Internet Applications (aka ARIA)
    - 另一个完全技能列表 https://platform.html5.org/
    - HTML
        - Most relevant specifications / documentation:
            * [All W3C HTML Spec](http://www.w3.org/standards/techs/html#w3c_all)
            * [The elements of HTML from the Living Standard](https://html.spec.whatwg.org/multipage)
            * [Global attributes](https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes)
            * [HTML 5.2 from W3C](http://w3c.github.io/html/)
            * [HTML attribute reference](https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes)
            * [HTML element reference](https://developer.mozilla.org/en-US/docs/Web/HTML/Element)
            * [The HTML Syntax](https://html.spec.whatwg.org/multipage/syntax.html#syntax) from the Living Standard
    - CSS
        - Most relevant specifications / documentation:
            * [All W3C CSS Specifications](http://www.w3.org/Style/CSS/current-work#roadmap)
            * [Cascading Style Sheets Level 2 Revision 2 (CSS 2.2) Specification](https://drafts.csswg.org/css2/)
            * [CSS reference](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference)
            * [Selectors Level 3](http://www.w3.org/TR/css3-selectors/)
    - DOM
        - Most relevant specifications / documentation:
            * [Document Object Model (DOM) Level 3 Events Specification](https://www.w3.org/TR/DOM-Level-3-Events/)
            * [DOM Living Standard](https://dom.spec.whatwg.org/)
            * [W3C DOM4](https://www.w3.org/TR/2015/REC-dom-20151119/)
    - JavaScript
        - Most relevant specifications / documentation:
            * [ECMAScript® 2017 Language Specification](https://tc39.github.io/ecma262/)
    - Web APIs (aka HTML5 and friends)
        - Most relevant documentation:
            * [Web API Interfaces](https://developer.mozilla.org/en-US/docs/Web/API)
    - Hypertext Transfer Protocol (aka HTTP)
        - Most relevant specifications:
            * [Hypertext Transfer Protocol -- HTTP/1.1](https://tools.ietf.org/html/rfc2616)
            * [HTTP/2](https://http2.github.io/)
    - Uniform Resource Locators (aka URL)
        -  A URL is a specific type of uniform resource identifier (URI).
        - Most relevant specifications:
            * [Uniform Resource Locators (URL)](http://www.w3.org/Addressing/URL/url-spec.txt)
            * [URL Living Standard](https://url.spec.whatwg.org/)
    - JavaScript Object Notation (aka JSON)
        - Most relevant specifications:
            * [Introducing JSON](http://json.org/)
            * [JSON API](http://jsonapi.org/)
            * [The JSON Data Interchange Format](http://www.ecma-international.org/publications/files/ECMA-ST/ECMA-404.pdf)
- Front-End Dev Skills
    * Content Management Systems (aka CMS)
    * Node.js
    * Cross-Browser Testing
    * Cross-Platform Testing
    * Unit Testing
    * Cross-Device Testing
    * Accessibility / WAI-ARIA
    * Search Engine Optimization (aka SEO)
    * Interaction or User Interface Design
    * User Experience
    * Usability
    * E-commerce Systems
    * Portal Systems
    * Wireframing
    * CSS Layout / Grids
    * DOM Manipulation (e.g., jQuery)
    * Mobile Web Performance
    * Load Testing
    * Performance Testing
    * Progressive Enhancement / Graceful Degradation
    * Version Control (e.g., GIT)
    * MVC / MVVM / MV*
    * Functional Programming
    * Data Formats (e.g., JSON, XML)
    * Data APIs (e.g Restful API)
    * Web Font Embedding
    * Scalable Vector Graphics (aka SVG)
    * Regular Expressions
    * Content Strategy
    * Microdata / Microformats
    * Task Runners, Build Tools, Process Automation Tools
    * Responsive Web Design
    * Object-Oriented Programming
    * Application Architecture
    * Modules
    * Dependency Managers
    * Package Managers
    * JavaScript Animation
    * CSS Animation
    * Charts / Graphs
    * UI Widgets
    * Code Quality Testing
    * Code Coverage Testing
    * Code Complexity Analysis
    * Integration Testing
    * Command Line / CLI
    * Templating Strategies
    * Templating Engines
    * Single Page Applications
    * XHR Requests (aka AJAX)
    * Web/Browser Security
    * HTML Semantics
    * Browser Developer Tools
- interviews
    - Questions you may get asked:
        * [10 Interview Questions Every JavaScript Developer Should Know](https://medium.com/javascript-scene/10-interview-questions-every-javascript-developer-should-know-6fa6bdf5ad95)
        * [Front-End Job Interview Questions](http://h5bp.github.io/Front-end-Developer-Interview-Questions/)
        * [Front End Web Development Quiz](http://davidshariff.com/quiz/)
        * [Interview Questions for Front-End-Developer](http://thatjsdude.com/interview/index.html)
        * [JavaScript Web Quiz](http://davidshariff.com/js-quiz/)
    - Questions you ask:
        * [An open source list of developer questions to ask prospective employers](https://github.com/ChiperSoft/InterviewThis)
    - Preparing:
        * [Preparing for a Front-End Web Development Interview in 2017](http://davidshariff.com/blog/preparing-for-a-front-end-web-development-interview-in-2017/)
        * [Interview Cake](https://www.interviewcake.com/) [$]
        * [Cracking the front-end interview](https://medium.freecodecamp.com/cracking-the-front-end-interview-9a34cd46237)


# part2 learning
- video materials
    * [codecademy.com](https://codecademy.com)
    * [codeschool.com](https://www.codeschool.com/)
    * [egghead.io](https://egghead.io/)
    * [eventedmind.com](https://www.eventedmind.com/)
    * [Frontend Masters](https://frontendmasters.com/)
    * [Freecodecamp](https://www.freecodecamp.com)
    * [Khan Academy](https://www.khanacademy.org/computing/computer-programming)
    * [laracasts.com](https://laracasts.com/)
    * [lynda.com](http://www.lynda.com/) [careful, quality varies]
    * [mijingo.com](https://mijingo.com/)
    * [pluralsight.com](http://www.pluralsight.com/) [careful, quality varies]
    * [Treehouse](https://teamtreehouse.com/)
    * [tutsplus.com](https://tutsplus.com/courses)
    * [Udacity](https://www.udacity.com/courses/web-development) [careful, quality varies]

- How Does the Internet Work?
    - http://web.stanford.edu/class/msande91si/www-spr04/readings/week1/InternetWhitepaper.htm
    - layers: application/TCP/IP/Harware
    - If the message to be sent is long, each stack layer that the message passes through may break the message up into smaller chunks of data.
    - In TCP layer, Each packet is assigned a port number.
    - IP layer, 添加IP地址
- learn web browsers
    -  The [most commonly used browsers](https://www.sitepoint.com/browser-trends-september-2016-browser-wars/) (on any device) are:
        1. [Chrome](http://www.google.com/chrome/) (engine: [Blink](https://en.wikipedia.org/wiki/Blink_%28layout_engine%29) + [V8](https://en.wikipedia.org/wiki/V8_%28JavaScript_engine%29))
        2. [Firefox](https://www.mozilla.org/en-US/firefox/new/) (engine: [Gecko](https://en.wikipedia.org/wiki/Gecko_%28software%29) + [SpiderMonkey](https://en.wikipedia.org/wiki/SpiderMonkey_%28software%29))
        3. [Internet Explorer](http://windows.microsoft.com/en-us/internet-explorer/download-ie) (engine: [Trident](https://en.wikipedia.org/wiki/Trident_%28layout_engine%29) + [Chakra](https://en.wikipedia.org/wiki/Chakra_%28JScript_engine%29))
        4. [Safari](https://www.apple.com/safari/) (engine: [Webkit](https://en.wikipedia.org/wiki/WebKit) + [SquirrelFish](https://trac.webkit.org/wiki/SquirrelFish))
    - The Most Commonly Used Headless Browser Are:
        * [PhantomJS](http://phantomjs.org/) (engine: [Webkit](https://en.wikipedia.org/wiki/WebKit) + SquirrelFish)
        * [SlimerJS](http://slimerjs.org/) (engine: [Gecko](https://en.wikipedia.org/wiki/Gecko_%28software%29) + [SpiderMonkey](https://en.wikipedia.org/wiki/SpiderMonkey_%28software%29))
        * [TrifleJS](http://triflejs.org/) (engine: [Trident](https://en.wikipedia.org/wiki/Trident_%28layout_engine%29) + [Chakra](https://en.wikipedia.org/wiki/Chakra_%28JScript_engine%29))
- DNS
  - wiki:The Domain Name System (DNS) is a hierarchical distributed naming system for computers, services, or any resource connected to the Internet or a private network. It associates various information with domain names assigned to each of the participating entities. Most prominently, it translates domain names, which can be easily memorized by humans, to the numerical IP addresses needed for the purpose of computer services and devices worldwide. The Domain Name System is an essential component of the functionality of most Internet services because it is the Internet's primary directory service.

- Learn HTTP/Networks  
    - HTTP Specifications
        * [HTTP/2](https://http2.github.io/)
        * [Hypertext Transfer Protocol -- HTTP/1.1](https://tools.ietf.org/html/rfc2616)
    - HTPP STATUS code
        - https://httpstatuses.com/
    - http book:  High Performance Browser Networking
        - content: networks/transport protocols/application protocols/apis(XHR websocket webrtc)

- Learn Web Hosting
    - https://www.whoishostingthis.com/resources/web-hosting/
- Learn General Front-End Development    
- Learn User Interface/Interaction Design
- Learn HTML & CSS
    - https://learn.shayhowe.com/advanced-html-css/complex-selectors/
    - https://learn.shayhowe.com/html-css/building-your-first-web-page/
    - https://code.tutsplus.com/tutorials/the-30-css-selectors-you-must-memorize--net-16048
    - https://developer.mozilla.org/en-US/docs/Web/JavaScript/A_re-introduction_to_JavaScript
- Learn DOM, BOM, & jQuery
    - document object model http://eloquentjavascript.net/13_dom.html
        -  document.documentElement as root
        - each DOM node object has a `nodeType `  property,指明节点类型,其实是枚举类, `document.**_NODE ` 是这些枚举类
        - all element node have a `getElementBytTagName getElementBytTagId getElementByClassId `method.
        - element returned by `querySelectorAll ` method is not live. It won't change when you change the document.
        - https://www.codecademy.com/courses/web-beginner-en-bay3D/0/1?curriculum_id=50a3fad8c7a770b5fd0007a1
        - book DOM Enlightenment: http://domenlightenment.com/
- Learn Web/Browser APIs        
    - HTML5: http://apress.jensimmons.com/v5/pro-html5-programming/ch0.html
    - api 文档: http://html5index.org/

- Learn Computer Science via JS    
    - https://github.com/nzakas/computer-science-in-javascript


- Learn Front-End Application Architecture    


- Learn Build and task automation
    - Build automation is the process of automating the creation of a software build and the associated processes including: compiling computer source code into binary code, packaging binary code, and running automated tests.

    
