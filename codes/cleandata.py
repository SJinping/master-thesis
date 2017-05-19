import csv
from operator import itemgetter

infile = '/home/pan/Idealab/Data/VA_Proc/emtion_tweets/survey/results_new/Batch_2640330_batch_results_python.csv'
with open(infile, 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
	header     = next(spamreader)
	
	headerid   = [int(item[12:]) for item in header[14:73:2]] # text ids
	scoreid_a  = [int(item.split('_')[1]) for item in header[74:133:2]]
	scoreid_v  = [int(item.split('_')[1]) for item in header[75:133:2]]
	
	question   = []
	arousal    = []
	valence    = []
	
	index      = 1
	for row in spamreader:
		workerid = row[1]

		try:
			question = zip([item for item in row[14:73:2]], headerid)
			arousal = zip(scoreid_a, [int(item) for item in row[74:133:2]])
			valence = zip(scoreid_v, [int(item) for item in row[75:133:2]])
		

			question.sort(key = itemgetter(0))
			arousal.sort(key = itemgetter(0))
			valence.sort(key = itemgetter(0))

			# find the replicated items
			pre = 0
			replicated = []
			for i in xrange(1, len(question)):
				if question[i][0] != question[pre][0]:
					if i-1 != pre: # more than one item
						replicated.append(zip(*question[pre:i])[1]) # pre to i-1 are the same
					pre = i

			ratings_a = []
			ratings_v = []
			for items in replicated:
				values_a = []
				values_v = []
				for item in items:
					cur = zip(*arousal)[0].index(item)
					values_a.append(zip(*arousal)[1][cur])

					cur = zip(*valence)[0].index(item)
					values_v.append(zip(*valence)[1][cur])
				ratings_a.append(tuple(values_a))
				ratings_v.append(tuple(values_v))

			print ('{}-worker: {}'.format(index, workerid))
			print ('arousal: {}'.format(ratings_a))
			print ('valence: {}'.format(ratings_v))

			if raw_input() == '\n':
				continue

		except Exception as e:
			continue

		index += 1
		
