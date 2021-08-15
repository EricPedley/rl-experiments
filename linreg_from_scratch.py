import random
a=2
b=3
noise_level = 1
data_length = 20
data = [(x,x*a+b+(random.random()-0.5)*noise_level) for x in range(data_length)]
#print(data)

weight = random.gauss(0,1)
bias = random.gauss(0,1)

learning_rate =  1e-4 #same as 0.0001 (1*10^(-4))

for epoch in range(100):
    for input,target in data:
        activation = input*weight+bias
        #loss = (activation-target)**2
        weight_grad = 2*(activation-target) * input
        bias_grad = 2*(activation-target)
        weight-=learning_rate*weight_grad
        bias-=learning_rate*bias_grad

print("Ground truth:",a,b)
print("Model estimates:",weight,bias)