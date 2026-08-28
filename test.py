y = 0
def myFunc():
  global y
  y = 10
  yield "Hello"
  y = 20
  yield 51

x = myFunc() # Generator Object

for z in x: 
    print("Y", y)
    print("Return ", z)