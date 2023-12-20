using CUDA

function addition!(x_d, y_d)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(x_d)
        @inbounds x_d[i] += y_d[i]
    end
    return nothing
end

function subtract!(x_d, y_d)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(x_d)
        @inbounds x_d[i] -= y_d[i]
    end
    return nothing
end

function multiply!(x_d, y_d)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(x_d)
        @inbounds x_d[i] *= y_d[i]
    end
    return nothing
end

function divide!(x_d, y_d)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(x_d)
        @inbounds x_d[i] /= y_d[i]
    end
    return nothing
end