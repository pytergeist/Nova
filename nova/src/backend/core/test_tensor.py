from nova.src.backend.core import Tensor, autodiff, grad_tape

autodiff.enabled(True)


t1 = Tensor([[1, 2, 3], [4, 5, 6], [5, 7, 9]], True)
t2 = Tensor([[1, 2, 3], [4, 5, 6], [5, 7, 9]], True)

t3 = t1 + t2
t3.backward()
print("start grad")
print(t3.get_grad().to_numpy())
print("end grad")
