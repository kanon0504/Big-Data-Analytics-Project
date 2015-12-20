import csv

def sort(a,b,c):
	pile = {'A':14,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
	categories = {'S':.8,'H':.6,'C':.4,'D':.2}
	hand = [a,b,c]
	numbers = [pile[i[0]]+categories[i[1]] for i in hand]
	order = [numbers.index(sorted(numbers)[i]) for i in range(len(numbers))]
	return [hand[order[0]],hand[order[1]],hand[order[2]]]

def rank(a,b,c,num):
	ordered_list =[]
	with open('ordered_combination.csv','r') as csvfile:
		csv_reader = csv.reader(csvfile,delimiter = ',')
		for i in csv_reader:
			temp = [(i[0],i[1]),(i[2],i[3]),(i[4],i[5])]
			ordered_list.append(temp)
	p = 1.-(ordered_list.index(sort(a,b,c))+1.0)/(len(ordered_list)+1.0)
	prob = p**num
	print 'Your winning chance against %d person is:'%num,prob

