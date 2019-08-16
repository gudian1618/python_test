# 列表
from sqlalchemy.sql.functions import random

a = range(1, 11, 2)
for i, j in enumerate(a):
	print(i, j)
	
# 字符串
s = 'dsafads'
for i, j in enumerate(s):
	print(i, j)

# 字典
k = {23, 'sadf', 213, 'ewwr'}
for i, j in enumerate(k):
	print(i, j)

for i in range(0, 3):
	print(i)


for j in xrange(0, 5):
	print(list(j))

