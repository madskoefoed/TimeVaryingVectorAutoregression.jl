num2mat(x::Real) = fill(x, 1, 1)
vec2mat(x::Vector) = reshape(x, :, 1)

get_ν(β) = β/(1 - β)
get_β(ν) = ν/(1 + ν)
get_k(p, β) = (β - p*β + p)/(2β - p*β + p - 1)