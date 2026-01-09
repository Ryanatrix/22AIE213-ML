import numpy #required for matrix operations 
from numpy.linalg import matrix_power #inbuilt funcyion already existing for matrix exponentiation
def sum(list1, sum): #function 1 that finds multiple combinations of numbers leading to the desired sum
    count1=0
    for i in range(len(list1)):
      for j in range(i+1,len(list1)):
         if list1[i]+list1[j]==sum:
            count1+=1
    return count1

def riyal(list2): # function 2 finds the minimum and maximum value by a 2 step comparison method
   min=list2[0]
   max=list2[0]
   for i in range(len(list2)-1):
      if list2[i+1]>list2[i]:
         if list2[i+1]>max:
            max=list2[i+1]
      elif list2[i+1]<list2[i]:
         if list2[i+1]<min:
            min=list2[i+1]
   return (max - min)

def matop(mat, exponent): # fuction 3 used an inbuilt function to exponentiate a matrix
   val=matrix_power(mat,exponent)
   return val

def charcount(string1): # function 4 counted the most repeated character
    freq = {}
    max_char = max(freq, key=freq.get)
    return max_char, freq[max_char]

def meanmedmod(arr1): # function 5 calculated mean mode and median for an array
    n = len(arr1)
    total = 0
    for i in arr1:
        total += i
    mean = total / n
    a = sorted(arr1)
    if n % 2 == 1:
        median = a[n // 2]
    else:
        median = (a[n // 2 - 1] + a[n // 2]) / 2

    freq = {}
    for x in arr1:
        freq[x] = freq.get(x, 0) + 1

    max_count = 0
    mode = None
    for k in freq:
        if freq[k] > max_count:
            max_count = freq[k]
            mode = k

    return mean, median, mode

def main(): # main function
   while True:
    print("welcome to this multi purpose multi function code written and developed by Rohan U for 22aie213 assignment 1")
    print("1. sum combinations")
    print("2. max - min")
    print("3. matrix exponentiation")
    print("4. string max frequency")
    print("5. mean median and mode")
    choice=int(input("enter choice of operation from 1 to 5: "))
    if choice==1:
        list1=[2,7,4,1,3,6]
        print(sum(list1,10))
    elif choice==2:
        list2=[5,3,8,1,0,4]
        print(riyal(list2))
    elif choice==3:
        mat=numpy.array([[3,4],[4,5]])   
        print(matop(mat,4))
    elif choice==4:
        string1="hippopotamus"
        print(charcount(string1))
    elif choice==5:
        arr1=[10, 20, 20, 30]
        print(meanmedmod(arr1))
    else:
        print("tough luck no more functions to try out hahaha")

main()


    