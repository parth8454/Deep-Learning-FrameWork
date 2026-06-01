import maths

class Value:
  
  def __init__(self,data,_children=(),_op=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __repr__(self):
    return f"Value(data={self.data},grad={self.grad})"

  def __add__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data + other.data, (self,other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __mul__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data * other.data,(self,other),'*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  def __pow__(self,other):
    assert isinstance(other,(int,float))
    out = Value(self.data**other,(self,),'**')

    def _backward():
      self.grad+=(other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out
  
  def __radd__(self, other): # Handles: other + self
        return self + other
  
  def __neg__(self):
    return self*-1
  
  def __sub__(self,other):
    return self + (-other)

  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
      
    build_topo(self)

    self.grad = 1.0

    for node in reversed(topo):
      node._backward()
