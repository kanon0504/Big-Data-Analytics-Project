# Big-Data-Analytics-Project
Big Data Analytics Project

下面看完可删
—————————————————————————————————————华丽丽的分割线—————————————————————————————————————————
看到这里有对lambda参数和reduceByKey的解释，你可以看一下：
首先，我们map一下

1 >>> wc = words.map(lambda x: (x,1))
2 >>> print wc.toDebugString()
3 (2) PythonRDD[3] at RDD at PythonRDD.scala:43
4 |  shakespeare.txt MappedRDD[1] at textFile at NativeMethodAccessorImpl.java:-2
5 |  shakespeare.txt HadoopRDD[0] at textFile at NativeMethodAccessorImpl.java:-2

>>> wc = words.map(lambda x: (x,1))

>>> print wc.toDebugString()
(2) PythonRDD[3] at RDD at PythonRDD.scala:43
|  shakespeare.txt MappedRDD[1] at textFile at NativeMethodAccessorImpl.java:-2
|  shakespeare.txt HadoopRDD[0] at textFile at NativeMethodAccessorImpl.java:-2

我使用了一个匿名函数（用了Python中的lambda关键字）而不是命名函数。这行代码将会把lambda映射到每个单词。因此，每个x都是一个单词，每个单词都会被匿名闭包转换为元组(word, 1)。为了查看转换关系，我们使用toDebugString方法来查看PipelinedRDD是怎么被转换的。可以使用reduceByKey动作进行字数统计，然后把统计结果写到磁盘。

>>> counts = wc.reduceByKey(add)
>>> counts.saveAsTextFile("wc")
————————————————————————————————————————————————————————————————————————————————————————————
