import csv
import numpy as np

def sort(a,b,c):
	pile = {'A':14,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
	categories = {'S':.8,'H':.6,'C':.4,'D':.2}
	hand = [a,b,c]
	numbers = [pile[i[0]]+categories[i[1]] for i in hand]
	order = [numbers.index(sorted(numbers)[i]) for i in range(len(numbers))]
	return hand[order[0]],hand[order[1]],hand[order[2]]


pile = {'A':14,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
numbers = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
categories = ['S','H','C','D']
deck = []
for number in numbers:
	for category in categories:
		deck.append((number,category))

dictionary = {}
for i in deck:
	for j in deck:
		if j != i:
			for k in deck:
				if (k != i and k != j):
					dictionary[sort(i,j,k)] = 0
for hand in dictionary:
	if hand[0][0]==hand[1][0]==hand[2][0]:
		dictionary[hand] += 1000000*pile[hand[0][0]]
	#Whole-colored
	elif hand[0][1]==hand[1][1]==hand[2][1] and \
	(pile[hand[2][0]]-pile[hand[1][0]]==pile[hand[1][0]]-pile[hand[0][0]]==1 or \
	(pile[hand[1][0]]-pile[hand[0][0]]==1 and pile[hand[2][0]]-pile[hand[0][0]]==12)):
		dictionary[hand]+=100000*pile[hand[0][0]] -pile[hand[2][0]]
	#Suit flush
	elif hand[0][1]==hand[1][1]==hand[2][1]:
		dictionary[hand]+=10000*pile[hand[2][0]]+1000*pile[hand[1][0]]+100*pile[hand[0][0]]
	#Suit
	elif pile[hand[2][0]]-pile[hand[1][0]]==pile[hand[1][0]]-pile[hand[0][0]]==1 or \
	(pile[hand[1][0]]-pile[hand[0][0]]==1 and pile[hand[2][0]]-pile[hand[0][0]]==12):
		dictionary[hand]+=1000*pile[hand[0][0]] -pile[hand[2][0]]
	#Straight
	elif hand[0][0]==hand[1][0]:
		dictionary[hand]+=100*pile[hand[1][0]]+pile[hand[2][0]]
	elif hand[1][0]==hand[2][0]:
		dictionary[hand]+=100*pile[hand[1][0]]+pile[hand[0][0]]
	#Pairs
	else:
		dictionary[hand]+=10*pile[hand[2][0]]+pile[hand[1][0]]+0.1*pile[hand[0][0]]
	#Regulars


ordered = []
i = len(dictionary)
while (i):
	print i
	keys = dictionary.keys()
	values = dictionary.values()
	ordered.append(keys[values.index(max(values))])
	del dictionary[ordered[len(ordered)-1]]
	i -= 1

with open('./ordered_combination.csv','w') as csvfile:
	writer = csv.writer(csvfile,delimiter = ',')
	for x in ordered:
		writer.writerow(list(np.concatenate(x)))
