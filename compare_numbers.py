a=input("Enter the value of a\n")
b=input("Enter the value of b\n")
c=input("Enter the value of c1\n")

if(a>b or a == b):
  if(a>b):
    print("a is greater")
elif(a==b):
  print("a is equal to b")
elif(b==c or b>c):
  if(b>c):
    print("b is greater")
  elif(b==c):
    print("b and c are equal")

elif(c>a or c == a):
  if(c>a):
    print("c is greater")
elif(c==a):
  print("c is equal to a")
else:
  print("Invalid")
