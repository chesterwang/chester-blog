---
title: python logging 模块探索 
date: 2019-11-25 00:47:41
categories: 工程开发系列
---


本文介绍了python logging 模块的大致原理。

<!-- more -->



目录

- [1. logging 体系结构](#1-logging-体系结构)
- [2. logging 中的数据结构](#2-logging-中的数据结构)
- [3. 源码阅读收获](#3-源码阅读收获)


# 1. logging 体系结构

1. Manager 管理Logger的树性体系
2. Logger 一个Logger实例可以包含多个Handler,底层调用self.callHandlers()方法来处理LogRecord
3. Handler 一个Handler可以包含多个,Logger.callHandlers()调用的是该类的handle方法
4. root为Logger树性体系的根logger,任何logger实例都挂靠在该logger(无论间接还是直接)


# 2. logging 中的数据结构

`logging` 模块内部的数据结构如下

```python
# 单条日志消息
class LogRecord(object):
    """
    A LogRecord instance represents an event being logged.
    """
    def getMessage(self):
```


```python
#日志格式风格

class PercentStyle(object):
    def format(self, record):

class StrFormatStyle(PercentStyle):

class StringTemplateStyle(PercentStyle):
```


```python
#格式化器
class Formatter(object):
    """
    Formatter instances are used to convert a LogRecord to text.
    def __init__(self, fmt=None, datefmt=None, style='%'):
    def formatMessage(self, record):
```

```python
#过滤器
class Filter(object):
    """
    Filter instances are used to perform arbitrary filtering of LogRecords.
    def filter(self, record):
    """

#过滤器集合
class Filterer(object):
    """
    A base class for loggers and handlers which allows them to share
    common code.
    """
```

```python
class Handler(Filterer):
    """
    Handler instances dispatch logging events to specific destinations.

    def handle(self, record):
        """
        Conditionally emit the specified logging record.

class StreamHandler(Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a stream. Note that this class does not close the stream, as
    sys.stdout or sys.stderr may be used.
    """

class FileHandler(StreamHandler):
    """
    A handler class which writes formatted logging records to disk files.
    """
```

```python
class Manager(object):
    """
    There is [under normal circumstances] just one Manager instance, which
    holds the hierarchy of loggers.
    """
    def getLogger(self, name):
    """
    Get a logger with the specified name (channel name), creating it
    if it doesn't yet exist. This name is a dot-separated hierarchical
    name, such as "a", "a.b", "a.b.c" or similar.
    """
```

```python
class Logger(Filterer):
    """
    Instances of the Logger class represent a single logging channel. A
    "logging channel" indicates an area of an application. Exactly how an
    "area" is defined is up to the application developer. Since an
    application can have any number of areas, logging channels are identified
    by a unique string. Application areas can be nested (e.g. an area
    of "input processing" might include sub-areas "read CSV files", "read
    XLS files" and "read Gnumeric files"). To cater for this natural nesting,
    channel names are organized into a namespace hierarchy where levels are
    separated by periods, much like the Java or Python package namespace. So
    in the instance given above, channel names might be "input" for the upper
    level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
    There is no arbitrary limit to the depth of nesting.
    """
```

# 3. 源码阅读收获

```python

# 类作为变量,构建实例
(self.loggerClass or _loggerClass)(name)

# 常用变量名
rv 意为 return value

# 将类看成是函数
_logRecordFactory = LogRecord 

# 更新各种实例的属性值
rv.__dict__.update(dict)
```

