'''a=[1,2,5,17,520,4399]
b=[]
def x():
    for i in a:
        if i%2==0:
            pass
        else:
            i=i*2
        b.append(i)
x()
print(b)'''
'''arr=[3,0,1,1,9,7]
a=7
b=2
c=3
for i in arr:
    for j in arr:
        for k in arr:
            if i<j and i<k and 0<i<len(arr) and 0<j<len(arr) and 0<k<len(arr) and (arr[i]-arr[j]<=a and arr[j]-arr[i]<=b) and (arr[i]-arr[k]<=c and arr[k]-arr[i]<=c) and (arr[j]-arr[k]<=b and arr[k]-arr[j]<=b):
                print((arr[i],arr[j],arr[k]))'''
'''class Plant:
    def __init__(self,name,age):
        self.__name = name
        self.age = age
    def get_name(self):
        return self.__name
    def color(self):
        pass
class flower(Plant):
    def color(self):
        return "Multiful"
class rose(flower):
    def color(self):
        return "red"
rose1=rose("wonderful rose",1)
print("the age of",rose1.color(),rose1.get_name(),"is",rose1.age)'''

'''def calculate():
    a=input("enter an int:")
    b=input("enter a signal:")
    c=input("enter an int:")
    format=a+b+c
    try:
        result=eval(format)
        a=int(a)
        c=int(c)
    except ValueError:
        print("Do not accept 'not int'" )
    except TypeError :
        print("Do not accept 'not int'" )
    except ZeroDivisionError:
        print("Do not accept '/0'" )
    except SyntaxError:
        print("Do not accept 'not signal'" )
    except NameError:
        print("What are you doing?" )
    else:
        print(result)
    return
print(calculate())'''







