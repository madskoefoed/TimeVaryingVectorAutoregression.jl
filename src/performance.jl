function goodness_of_fit(tvvar::TVVAR)

    denom = collect(size(tvvar.e, 1))

    ME = cumsum(tvvar.e; dims = 1) ./ denom
    MAE = cumsum(abs.(tvvar.e); dims = 1) ./ denom
    RMSE = sqrt.(cumsum(tvvar.e.^2; dims = 1) ./ denom)
    MSSE = cumsum(tvvar.u.^2; dims = 1) ./ denom

    return (ME = ME, MAE = MAE, RMSE = RMSE, MSSE = MSSE)

end