notes on model at https://github.com/ritheshkumar95/pytorch-vqvae

### Resblock

- relu
- relu(bn(conv, kernel_size=3, stride=1, padding=1) 
- bn(conv, kernel_size=1, stride=1, padding=0)
- x + block(x)

### Encoder

- relu(bn(conv, dim filters, filter_size=4, stride=2, padding=1)
- conv, dim filters, filter_size=4, stride=2, padding=1
- resblock
- resblock

### Decoder

- resblock
- resblock
- relu(bn(convtranspose, dim filters, kernel_size=4, stride=2, padding=1))
- tanh(convtranspose, input_dim filters, kernel_size=4, stride=2, padding=1)

### VQVAE


