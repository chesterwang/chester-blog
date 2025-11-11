---
title: spark dataframe操作总结!
date: 2016-07-16
tags: [spark]
---

spark datagrame的操作总结

### 资料
- https://databricks.com/blog/2016/01/04/introducing-apache-spark-datasets.html
    - Spark Datasets, an extension of the DataFrame API that provides a type-safe, object-oriented programming interface.
    - Datasets extend these benefits with compile-time type safety – meaning production applications can be checked for errors before they are run. 这点非常重要，因为在这种大数据应用开发中，构造测试数据和运行测试程序都比较麻烦，有了type-safe就可以省去很多麻烦。
    - Both APIs make it easy to express the transformation using lambda functions.
    - with Datasets you also have access to all the power of a full relational execution engine.
- http://www.agildata.com/apache-spark-rdd-vs-dataframe-vs-dataset/
    -  The DataFrame API introduces the concept of a schema to describe the data, allowing Spark to manage the schema and only pass data between nodes, in a much more efficient way than using Java serialization.
    - Because the code is referring to data attributes by name, it is not possible for the compiler to catch any errors. If attribute names are incorrect then the error will only detected at runtime, when the query plan is created.
    - the familiar object-oriented programming style and compile-time type-safety of the RDD API but with the performance benefits of the Catalyst query optimizer.
    - Additionally, the Dataset API is designed to work equally well with both Java and Scala. When working with Java objects, it is important that they are fully bean-compliant
- https://databricks.com/blog/2015/08/12/from-pandas-to-apache-sparks-dataframe.html
    - a few operations that you can do in Pandas don’t translate to Spark well
    - 不可像pandas那样运用行列坐标获取单个元素
    - 不能 col1 = col2\*col3这种形式，因为immutable.
    - The only difference is that in Pandas, it is a mutable data structure that you can change – not in Spark.
    - 变态的列命名: In Spark SQL DataFrame columns are allowed to have the same name, they’ll be given unique names inside of Spark SQL, but this means that you can’t reference them with the column name only as this becomes ambiguous.

### import

    //val sqlContext = new SQLContext(sc)
    //import sqlContext.implicits._

### DataFrame creation

    //createDataFrame
    val df = sqlContext.createDataFrame(Array((3,8,"asdf"),(4,8,"ewe"),(4,8,"-pl["))).toDF("a","b","c")
    df.show(10)

    //rdd(parallelize) then tranform to DF using toDF
    val rdd = sc.parallelize(Seq((1,2),(3,4),(3,5)))
    val df2 = rdd.toDF("name","age")// this creates a DataFrame with column name "name" and "age"
    df2.show(10)
    val df3 = rdd.toDF()  // this implicit conversion creates a DataFrame with column name _1 and _2
    df3.show(10)

    // load method(depreacated)
    val df4 = sqlContext.load("com.databricks.spark.csv",
      Map("path" -> "/home/chester/data/2016-06/partsorthead", "header" -> "true", "delimiter" -> "\t"))
    df4.show(10)

    // read
    val options = Map("header" -> "true", "path" -> "/home/chester/data/2016-07/dist")
    val newStudents = sqlContext.read.options(options).format("com.databricks.spark.csv").load()
    newStudents.show(10)

### save to hdfs

	df.select("year", "model").save("newcars.csv", "com.databricks.spark.csv")

### column rename

    withColumnRenamed(existingName: String, newName: String): DataFrame

### single column operation

    //Returns a new DataFrame by adding a column or replacing the existing column that has the same name.
    //withColumn(colName: String, col: Column): DataFrame 
    //(column => column(new value)) or (column => new column(new value))
    val clickSeqUdf = udf((score: String) => score.replace("(","").toInt-1)
    val newdf2 = newdf.withColumn("clickSeq",clickSeqUdf(newdf.col("clickSeq")))

### multiple column operation ((column1,column2) => new column(new value)))

    //udf definition
    udf[ouput type, input2 type, input2 type]

### udf definition and application

    val toInt    = udf[String, String]( _+ "haha")
    val newdf = df.withColumn("name",(udf[String, String]( _+ "haha")).apply(df.col("name")))

### DataFrame join

    df1.join(department, df1("deptId") === department("id")

### udf of different style

    //显示定义udf，无类型推断
    val checksum = udf[Int,String,Int](_.toInt+_)
    df.withColumn("name",checksum.apply($"name",$"age")).show(10)

    //显示定义udf函数,类型推断
    val a:(Int,Int) => Int =(x:Int,y:Int) => x+y
    //简略定义
    val b = (x: String, y: Int) => x.toInt + y
    df.withColumn("name", udf(b).apply($"name", $"age")).show(10)

    //隐式定义udf函数,自动类型推断
    df.withColumn("name", udf((x: String, y: Int) => x.toInt + y).apply($"name", $"age")).show(10)
    df.filter(udf((x: String, y: Int) => x.toInt-100 > y).apply($"name", $"age")).show(10)

    //隐式定义，无类型推断
    df.withColumn("name", udf[String, String]( x =>  x+ "haha").apply($"name")).show(10)

    //org.apache.spark.sql.functions里定义了大量的udf

### other operation
    
如果DataFrame的method无法满足需求的时候，可以调用rdd()转化为rdd, 在rdd上做更灵活的操作，之后再转化为DataFrame.
