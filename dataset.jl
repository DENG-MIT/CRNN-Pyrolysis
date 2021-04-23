
const l_exp = 1:4
n_exp = length(l_exp)

l_train = []
l_val = []
for i = 1:n_exp
    j = l_exp[i]
    if !(j in [3])
        push!(l_train, i)
    else
        push!(l_val, i)
    end
end

l_exp_data = [];
l_exp_info = zeros(Float64, length(l_exp), 3);
l_exp_info[:, 2] = [5.0, 10.0, 20.0, 30.0];
l_exp_info[:, 3] .= -100.0;

l_fname = ["1.Beechwood-05.xlsx", "2.Beechwood-10.xlsx", 
           "3.Beechwood-20.xlsx", "4.Beechwood-30.xlsx"]

for (i_exp, value) in enumerate(l_exp)
    fname = l_fname[i_exp]
    xf = XLSX.readxlsx("./results/$fuel_name/exp_data/$fname")
    exp_data = Float64.(xf[1][:])
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    exp_data[:, 1] .+= 273.15

    l_exp_info[i_exp, 1] = exp_data[1, 1] # initial temperature, K
    
    ind = findmax(exp_data[:, 1])[2]
    exp_data_ = zeros(ind, 3)
    exp_data_[:, 1] .= (exp_data[1:ind, 1] .- exp_data[1, 1]) ./ l_exp_info[i_exp, 2] .* 60.0
    exp_data_[:, 2] .= exp_data[1:ind, 1]
    exp_data_[:, 3] .= exp_data[1:ind, 2] ./ exp_data[1, 2]

    push!(l_exp_data, exp_data_)
end
