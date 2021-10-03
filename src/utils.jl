get_ν(β) = β/(1 - β)
get_β(ν) = ν/(1 + ν)
get_k(p, β) = (β - p*β + p)/(2β - p*β + p - 1)
get_d(setup::Setup) = convert(Integer, (size(setup.m, 1) - 1)/size(setup.m, 2))