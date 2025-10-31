'''num=int(input("Enter a number:"))
if num<0:
    print("Negative number")
elif num>0:
    print("Positive number")
else:
    print("No number")'''
'''x=1
while x<10:
    print(x)
    x+=1'''
'''for x in range(2,101,5):
    print(x)'''
woeds =['this','sdd','sss']
'''for word in woeds:
    print(word)'''
'''def show():
    print('ggg')
    print('ddd')
    print('dddd')
show()
def show():
    print('hhh')'''
'''def h(str,num):
    print(str,num)
    return
h(num='蓝山',str=1)'''
'''def printinfo(name,age=18):
    print(name)
    print("Age:",age)
    return
printinfo(age=50,name='mik')
printinfo(name='miki')'''
#def mih(a,*args):
#    print(a)
#    for ar in args:
#        print(ar)
#    return
#mih( 10 )
#mih(10,20,30,40)

#sum=lambda arg1,arg2,amm:arg1+arg2*amm
#num=lambda:666
#print("相加后的值为：",sum(10,20,2))
#print(num())
'''total=0
def sum(arg1,arg2):
    total=arg1+arg2
    print("函数内是局部变量:",total)
    return total
sum(10,20)
print("函数外是全局变量：",total)'''
'''def get_multi(arg1,arg2):
    total=arg1+arg2
    return total
print(get_multi(10,20))
print('-'*50)'''
'''def sum(a,b):
    ret1=a+b
    ret2=a-b
    return ret1,ret2
result=sum(10,20)
print(result)'''
'''def show(a,b):
    print("今天你PY了吗")
    return a,b
print(show(1,2))
help(show)'''
'''def func():
    print('py')
def test():
    print(111)
    func()
    print(222)
test()'''
'''def line():
    print("2"*20)
def linen(a):
    i=0
    while i<a:
        line()
        i+=1
linen(2)'''
#def amm(a:int,b:int) ->int:
#    return a+b
#a=int(input("a="))
#b=int(input("b="))
#print("a+b={}".format(amm(a,b)))
'''class Animall:
    def __init__(selff,name,age):
        selff.name=name
        selff.age=age
    def make_soundd(selff):
        pass
class Cat(Animall):
    def make_soundd(selff):
        return "Meow"
class Dog(Animall):
    def make_soundd(selff):
        return "Woof"
cat1=Cat("Kitty",3)
dog1=Dog("Buddy",5)
print(cat1.name,"says:",cat1.make_soundd())
print(dog1.name,"says:",dog1.make_soundd())
print(cat1.name,"My age is:",cat1.age)'''
'''class Car:
    num_cars=0
    def __init__(self,make,year):
        self.make=make
        self.year=year
    def add(self,onn):
        self.num_cars+=onn
    def selecet(self,ouu):
        self.num_cars-=ouu
car1=Car("dazong",1999)
print(car1.num_cars)
car1.add(50)
print(car1.num_cars)
car1.selecet(10)
print(car1.num_cars)'''
'''class Dog:
    num_of_dogs=0
    def __init__(self,name):
        self.name=name
        Dog.num_of_dogs+=1
    @classmethod
    def get_num_of_dogs(cls):
        return Dog.num_of_dogs
dog1=Dog("Buddy")
dog2=Dog("Kitty")
print(Dog.get_num_of_dogs())'''
'''class Math:
    @staticmethod
    def add(x,y):
        return x+y
result=Math.add(1,2)
print(result)'''
'''class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def get_name(self):
        return self.name
person1 = Person("Tiyong", 30)
print(person1.name)
print(person1.get_name())'''
#class Person:
#    def __init__(self, name, age):
#        self.__name = name
#        self.__age = age
#    def get_name(self):
#        return self.__name
#    def set_name(self,new_name):
#        self.__name=new_name
#    def display_info(self):
#        return f"Name:{self.__name},Age:{self.__age}"
#person1=Person("Tiyong",30)
#print(person1.display_info())
#person1.set_name("Toy")
#print(person1.display_info())

'''class ParentClass:
    def parent(self):
        print("Parent Class")
class ChildClass(ParentClass):
    def child(self):
        print("Child Class")
child=ChildClass()
child.parent()
child.child()'''
'''class PersonClass1:
    def method1(self):
        print("method1")
class PersonClass2:
    def method2(self):
        print("method2")
class ChildClass(PersonClass1,PersonClass2):
    def child_method(self):
        print("method")
child=ChildClass()
child.method1()
child.method2()
child.child_method()'''
#class Animal:
#    def speak(self):
#        raise NotImplementedError("Subclass must implement abstract method")
#class Dog(Animal):
#    def speak(self):
#        return "Woof!"
#class Cat(Animal):
#    def speak(self):
#        return "Meow"
#def animal_sound(animal):
#    return animal.speak()
#dog=Dog()
#cat=Cat()
#print(animal_sound(dog))
#print(animal_sound(cat))
'''def print_info(str,*aegs):
    print(str)
    for x in aegs:
        print(x)
    return
print_info(10)
print_info(10,20,30)'''
'''class Animal:
    def speak(self):
        return 0
class Dog(Animal):
    def speakl(slf):
        print("Dog")
class Cat(Animal):
    def speak(self):
        return "Meow!"
def animal_sound(animal):
    return animal.speak()
dog = Dog()
cat = Cat()
print(animal_sound(dog))
print(animal_sound(cat))'''
'''class Calculator:
    def add(self,a,b):
        return a+b
    def add(self,a,b,c):
        return a+b+c'''
'''import time
time.sleep(1)
print(111)'''
'''from time import sleep
sleep(1)
print(111)'''
'''from time import sleep as ah
ah(1)
print(111)'''
'''import ganzhi
ganzhi.m(1,0)'''
'''from ganzhi import *
m(0,1)'''

'''class Error(Exception):
    pass
def functionName(x):
    if x<1:
        raise Exception("error",x)
try:
    functionName(0)
except Error as e:
    print(e)'''
'''str1="hello"
str2="world"
result=(str1+"wo"+str2)
print(result)
slice=str1[2:3]#左闭右开
print(slice)
index=result.find("world")
print(index)
newresult =result.replace("world",slice)
print(newresult)
new_str=result.split("llo")
print(new_str)'''
'''name="Alice"
age=18
formatted_str="My name is {} and I am {} years old.".format(name,age)
print(formatted_str)
str = "         Hello World         "
upper_str=str.upper()
print(upper_str)
lowerstr=str.lower()
print(lowerstr)
_str=str.strip()
print(_str)'''
'''str="1Hello World"
starts_str=str.startswith("Hello")
ends_str=str.endswith("World")
is_digit=str.isdigit()
print(starts_str)
print(ends_str)
print(is_digit)'''
'''l=[1,2,3,4,5,6]
l.insert(0,5)
l.remove(4)
print(l)
l.sort()
print(l)
l_copy=l.copy()
print(l_copy)'''
'''my_dict = {"name":"Alice","age":18,"gender":"male"}
print(my_dict)
my_dict["age"]=19
print(my_dict)
del my_dict["age"]
print(my_dict)
print(len(my_dict))
print(my_dict.keys())
for key in my_dict.keys():
    print(key)'''
































