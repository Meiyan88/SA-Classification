function D = UpdateD(V)

d = 1 ./ sqrt(sum(V .^ 2, 2) + eps);
D =  diag(0.5 * d);
end