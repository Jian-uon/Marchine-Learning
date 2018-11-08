
## 数据库中的空值与NULL的区别以及python中的NaN和None

· None是一个python特殊的数据类型， 但是NaN却是用一个特殊的float

· 往数据库中写入时NaN不可处理，需转换成None,否则会报错。

· NULL(数据库)=None(python列表)=NaN(pandas)

· 空字符(数据库)=空字符(python列表)=空字符(pandas)

· 从csv中获取数据时：空值(csv)=NULL(数据库)=NaN(pandas)

· 转为csv数据时：数据库中的NULL\空字符和pandas中的NaN\空字符，都变成csv中的空值
