if __name__ == '__main__':
	print('请输入数字0进行训练，输入数字1进行写诗歌')
	number = input(' 输入一个数字吧：' )
	if isinstance(int(number), int):
		if int(number) == 0:
			print('Hello, 这是个0')
		else:
			print("这是个整数")
	else:
		print(" 小哥哥，你输入的有些东西哟，要不要输入个数字试试看")