from engine.value import Value

def test_for_backprop():
    
    a = Value(2.0)
    b = Value(4.0)
    c = Value(10.0)

    f = a * b + c

    f.backward()

    print(f.data)
    print(a.grad)
    print(b.grad)
    print(c.grad)

if __name__ == "__main__":
    test_for_backprop() 