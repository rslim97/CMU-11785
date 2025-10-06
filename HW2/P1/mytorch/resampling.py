import numpy as np

class Upsample1d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, x):
        N, C, input_width = x.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1
        z = np.zeros(shape=(N, C, output_width),dtype=np.float64)
        for i in range(0, output_width, self.upsampling_factor):
            z[:, :, i] = x[:, :, i // self.upsampling_factor]
        return z

    def backward(self, dLdz):
        N, C, output_width = dLdz.shape
        input_width = (output_width - 1) // self.upsampling_factor + 1
        dLdx = np.zeros(shape=(N, C, input_width),dtype=np.float64)
        for i in range(0, output_width, self.upsampling_factor):
            dLdx[:, :, i // self.upsampling_factor] = dLdz[:, :, i]
        return dLdx

class Downsample1d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, x):
        N, C, input_width = x.shape
        self.original_input_width = input_width
        output_width = (input_width - 1) // self.downsampling_factor + 1
        z = np.zeros(shape=(N, C, output_width),dtype=np.float64)
        for i in range(output_width):
            z[:, :, i] = x[:, :, i * self.downsampling_factor]
        return z

    def backward(self, dLdz):
        N, C, output_width = dLdz.shape
        # input_width = (output_width - 1) * self.downsampling_factor + 1
        # if output_width % 2 == 1:
        #     print("output_width ", output_width)
        #     print("input_width ", input_width)
        #     print("self.original ", self.original_input_width)
        #     # when output_width is odd, input_width equals self.original_input_width
        #     assert input_width == self.original_input_width
        dLdx = np.zeros(shape=(N, C, self.original_input_width),dtype=np.float64)
        for i in range(output_width):
            dLdx[:, :, i * self.downsampling_factor] = dLdz[:, :, i]
        return dLdx

class Upsample2d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, x):
        N, C, input_height, input_width = x.shape
        output_height = (input_height - 1) * self.upsampling_factor + 1
        output_width = (input_width - 1) * self.upsampling_factor + 1
        z = np.zeros(shape=(N, C, output_height, output_width),dtype=np.float64)
        for i in range(0, output_height, self.upsampling_factor):
            for j in range(0, output_width, self.upsampling_factor):
                z[:, :, i, j] = x[:, :, i // self.upsampling_factor, j // self.upsampling_factor]
        return z

    def backward(self, dLdz):
        N, C, output_height, output_width = dLdz.shape
        input_height = (output_height - 1) // self.upsampling_factor + 1
        input_width = (output_width - 1) // self.upsampling_factor + 1
        dLdx = np.zeros(shape=(N, C, input_height, input_width),dtype=np.float64)
        for i in range(0, output_height, self.upsampling_factor):
            for j in range(0, output_width, self.upsampling_factor):
                dLdx[:, :, i // self.upsampling_factor, j // self.upsampling_factor] = dLdz[:, :, i, j]
        return dLdx

class Downsample2d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, x):
        N, C, input_height, input_width = x.shape
        self.original_input_height = input_height
        self.original_input_width = input_width
        output_height = (input_height - 1) // self.downsampling_factor + 1
        output_width = (input_width - 1) // self.downsampling_factor + 1
        z = np.zeros(shape=(N, C, output_height, output_width),dtype=np.float64)
        for i in range(output_height):
            for j in range(output_width):
                z[:, :, i, j] = x[:, :, i * self.downsampling_factor, j * self.downsampling_factor]
        return z

    def backward(self, dLdz):
        N, C, output_height, output_width = dLdz.shape
        dLdx = np.zeros(shape=(N, C, self.original_input_height, self.original_input_width),dtype=np.float64)
        print("sampling dLdx.shape ", dLdx.shape)
        print("sampling dLdz.shape ", dLdz.shape)
        for i in range(output_height):
            for j in range(output_width):
                dLdx[:, :, i * self.downsampling_factor, j * self.downsampling_factor] = dLdz[:, :, i, j]
        return dLdx

if __name__ == "__main__":
    ud = Upsample1d(2)
    x = np.array([[[1, 0, -1, 2, 1]]])
    z = ud.forward(x)
    dLdz = ud.backward(z)
    print(z)
    print(dLdz)

    ds = Downsample1d(2)
    # x = np.array([[[1, 0, 1, 2, -1, 5, 3]]])
    # x = np.array([[[1, 0, 1, 2, -1, 5]]])
    # x = np.array([[[1,2,3,4,5,6,7,8,9]]])
    x = np.array([[[1,2,3,4,5,6,7,8,9,10]]])
    z = ds.forward(x)
    dLdz = ds.backward(z)
    print(z)
    print(dLdz)

    ud2d = Upsample2d(2)
    x = np.array([[[[0,1],[2,3]]]])
    z = ud2d.forward(x)
    dLdz = ud2d.backward(z)
    print(z)
    print(dLdz)
    x1 = x
    ud2d_1 = Upsample2d(3)
    z1 = ud2d_1.forward(x1)
    dLdz1 = ud2d_1.backward(z1)
    print(z1)
    print(dLdz1)

    ds2d = Downsample2d(3)
    x = np.random.randint(0,5,(1,1,4,4))
    z = ds2d.forward(x)
    print(x)
    print(z)
    dLdz = ds2d.backward(z)
    print(dLdz)
