class Person:
    def __init__(self):
        self._age = 0
 
    @property
    def age(self):           # getter
        return self._age
 
    @age.setter
    def age(self, value):    # setter
        self._age = value
 
james = Person()
james.age = 20      # 인스턴스.속성 형식으로 접근하여 값 저장
print(james.age)

# class Person:
#     def __init__(self):
#         self._age = 0
 
#     def get_age(self):           # getter
#         return self._age
 
#     def set_age(self, value):    # setter
#         self.e = value
 
# a = Person()
# a.set_age = 20
# print(a.get_age())