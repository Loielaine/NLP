import pandas as pd



def transform_data(file_dir,file_name):
	df = pd.read_csv(file_dir+file_name)
	df['text'] = df[['sentence1', 'sentence2','sentence3', 'sentence4','sentence5']].agg(' '.join,
	                                                                                                axis = 1)
	text = df['text'].values
	with open(file_dir+"line_"+file_name,'w') as writer:
		for t in text:
			writer.write(t+'\n')
	writer.close()
	return 0

transform_data( "/Users/Loielaine/Desktop/umich-2020/SI630/project/data/", 'ROCStories2017.csv')
