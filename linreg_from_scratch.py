import random
a=2
b=3
noise_level = 1
data_length = 20
data = [(x,x*a+b+(random.random()-0.5)*noise_level) for x in range(data_length)]
#print(data)

weight = random.gauss(0,1)
bias = random.gauss(0,1)

learning_rate =  1e-3 #same as 0.001 (1*10^(-3))

num_epochs=1000

for epoch in range(num_epochs):
    for input,target in data:
        activation = input*weight+bias
        weight_grad = 2*(activation-target) * input
        bias_grad = 2*(activation-target)
        weight-=learning_rate*weight_grad
        bias-=learning_rate*bias_grad

loss_sum=0

for input,target in data:
    loss = (input*weight+bias-target)**2
    loss_sum+=loss

print("Average loss:",loss_sum/data_length)