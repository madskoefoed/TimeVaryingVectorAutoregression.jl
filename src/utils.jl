
num2mat(x::Real) = fill(x, 1, 1)
vec2mat(x::Vector) = repeat(x, 1, 1)

get_ν(β) = β/(1 - β)
get_β(ν) = ν/(1 + ν)
get_k(p, β) = (β - p*β + p)/(2β - p*β + p - 1)

function goodness_of_fit(estim::Estimation)
    ME = mean(estim.e; dims = 1)
    MAE = mean(abs.(estim.e); dims = 1)
    RMSE = sqrt.(mean(estim.e.^2; dims = 1))
    MSSE = mean(estim.u.^2; dims = 1)
    return (ME = ME, MAE = MAE, RMSE = RMSE, MSSE = MSSE)
end

function goodness_of_fit(estim::Estimation)
    ME = mean(estim.e; dims = 1)
    MAE = mean(abs.(estim.e); dims = 1)
    RMSE = sqrt.(mean(estim.e.^2; dims = 1))
    MSSE = mean(estim.u.^2; dims = 1)
    return (ME = ME, MAE = MAE, RMSE = RMSE, MSSE = MSSE)
end