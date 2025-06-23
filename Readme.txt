本项目中包含HNSW算法的实现和简易的前后端服务，使用pycharm或vscode打开均可

由于数据集和存储图结构较大，压缩包中未上传， 数据集地址：https://www.kaggle.com/datasets/splcher/animefacedataset

目录结构如下：

HNSW/
├─ archieve/  			# 存放数据集图片	               		
├─ hnsw/               		
│  ├─ HNSW.py     		# 构建HNSW分层图
│  ├─ ImageVectorizer.py 	# 图片向量化
│  └─ Recommender.py	# 检索器负责 
├─ output/  				# 存放HNSW图结构
├─ web/                 		# 图片检索系统前后端	
│  ├─ static         			
│     └─ uploads/			# 暂存上传文件    
│  └─ templates           		# 前端HTML文件
│     ├─ upload.html        	# 提交页面
│     └─ results.html		# 结果页面       
└─ Readme.txt              

下载数据集后，在HNSW.py中修改数据集目录，运行HNSW.py生成HNSW.py，会自动保存到output目录下，

构建HNSW分层图后，修改web中数据和模型目录运行即可