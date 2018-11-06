# 
import time

#开始时间
start_time = time.clock()

from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io

if __name__ == '__main__':
	print('请输入数字0进行训练，输入数字1进行写诗歌')
	number = input(' 输入一个数字吧：' )
	if isinstance(int(number), int):
		if int(number) == 0:
			#结束时间
			end_time = time.clock()

			#计算时差
			minium = end_time - start_time

			print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")

			print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

			model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
		else:
			generate_output()
	else:
		print(" 小哥哥，你输入的有些东西哟，要不要输入个数字试试看")