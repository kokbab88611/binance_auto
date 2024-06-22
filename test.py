class a:
    def __init__(self) -> None:
        self.aa = 3

    def printself(self):
        print(self.aa)


class b:
    def __init__(self) -> None:
        pass

    def change(obj):
        print(obj.aa)

at = a()
b.change(at)
